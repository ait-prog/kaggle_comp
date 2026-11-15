# dev7 v2 (v1 went NaN at step ~410 in fp16, then OOM at eval): fp16 clamp hooks, non-finite-step skip + rollback, lr_a 1e-4, eval bs 8.
# dev7: Qwen3-8B-base (truncated) LoRA + multi-layer mean-pool head, trained on the curated open pool (KGR) on 2xT4 via DDP.
# Curation (CPU harness, +0.030 BA [+0.016,+0.044] vs raw): drop Benchmarks-proxy rows after the host ruling,
# gemini-3.7 from Vertex sources only, kimi-k3 reasoning=high only. Eval: all 2,699 official train rows (never trained on).
import os, sys, json, time, math, glob
import numpy as np, pandas as pd

CFG = dict(run='dev7', hidden=4096, n_layers=36, keep=24, layers=[4, 8, 12, 16, 20, 24], r=16, alpha=32, ldrop=0.05,
           cap=3000, epochs=1, bs=8, maxlen=320, lr_a=1e-4, lr_h=1e-3, wd=0.01, warm=0.05, seed=0, ckpt_every=800)
OUT = '/kaggle/working/out'

def load_data():
    comp = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)
    tr = pd.read_csv(comp)
    pool = pd.read_parquet(next(iter(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True))),
                           columns=['text', 'model', 'source_dataset', 'gen_route', 'gen_reasoning', 'flag_kb_proxy_after_ruling'])
    m = ~pool.flag_kb_proxy_after_ruling.values
    vert = (pool.source_dataset.str.contains('vertex', case=False) | pool.gen_route.str.contains('vertex', case=False)).values
    m &= (pool.model.values != 'gemini-3.7-flash') | vert
    m &= (pool.model.values != 'kimi-k3') | (pool.gen_reasoning.values == 'high')
    pool = pool[m].reset_index(drop=True)
    labels = sorted(tr.model.unique())
    rng = np.random.RandomState(CFG['seed']); idx = []
    for l in labels:
        w = np.flatnonzero((pool.model == l).values)
        idx.append(rng.choice(w, min(len(w), CFG['cap']), replace=False))
    pool = pool.iloc[np.concatenate(idx)].reset_index(drop=True)
    return tr, pool, labels

def find_model():
    for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try: j = json.load(open(c))
        except Exception: continue
        if j.get('model_type') == 'qwen3' and (j.get('hidden_size'), j.get('num_hidden_layers')) == (CFG['hidden'], CFG['n_layers']):
            return os.path.dirname(c)
    raise SystemExit('model not found')

def main(rank, world):
    import torch, torch.distributed as dist
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29512')
    import datetime; dist.init_process_group('nccl', rank=rank, world_size=world, timeout=datetime.timedelta(hours=1))
    torch.cuda.set_device(rank); dev = torch.device('cuda', rank)
    T0 = time.time()
    def log(*a):
        if rank == 0: print(f'[{time.time() - T0:6.0f}s]', *a, flush=True)
    torch.manual_seed(CFG['seed'])
    tr, pool, LABELS = load_data(); lid = {l: i for i, l in enumerate(LABELS)}
    log('cfg', json.dumps(CFG)); log('pool', len(pool), pool.model.value_counts().to_dict())
    MD = find_model(); log('model', MD)
    tok = AutoTokenizer.from_pretrained(MD); EOS = tok.eos_token_id
    PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
    enc = lambda t: [x + [EOS] for x in tok(list(t), add_special_tokens=False, truncation=True, max_length=CFG['maxlen'] - 1)['input_ids']]
    P_ids, P_y = enc(pool.text.values), pool.model.map(lid).values
    T_ids, T_y = enc(tr.text.values), tr.model.map(lid).values
    log('tokenized; pool p50/p99 len', int(np.median([len(x) for x in P_ids])), int(np.percentile([len(x) for x in P_ids], 99)))

    class LoRA(nn.Module):
        def __init__(self, base, r, alpha, p):
            super().__init__()
            self.base = base
            self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
            self.s = alpha / r; self.drop = nn.Dropout(p)
        def forward(self, x):
            return self.base(x) + ((self.drop(x) @ self.A.t().to(x.dtype)) @ self.B.t().to(x.dtype) * self.s)

    mc = AutoConfig.from_pretrained(MD); mc.num_hidden_layers = CFG['keep']; mc.use_cache = False
    if getattr(mc, 'layer_types', None) is not None: mc.layer_types = mc.layer_types[:CFG['keep']]
    base = AutoModel.from_pretrained(MD, config=mc, torch_dtype=torch.float16, attn_implementation='sdpa', device_map={'': rank}, low_cpu_mem_usage=True)
    for p in base.parameters(): p.requires_grad_(False)
    for layer in base.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
                if hasattr(parent, n): setattr(parent, n, LoRA(getattr(parent, n), CFG['r'], CFG['alpha'], CFG['ldrop']))
    def _clamp(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.dtype == torch.float16: h = torch.nan_to_num(h, nan=0.0, posinf=6e4, neginf=-6e4).clamp_(-6e4, 6e4)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    for layer in base.layers: layer.register_forward_hook(_clamp)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    H, L = mc.hidden_size, CFG['layers']

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = base
            self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in L])
            self.drop = nn.Dropout(0.1); self.fc = nn.Linear(H * len(L), len(LABELS))
        def feats(self, x, m):
            hs = self.base(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False).hidden_states
            mm = m.unsqueeze(-1).float(); d = mm.sum(1)
            return torch.cat([nm((hs[l].float() * mm).sum(1) / d) for l, nm in zip(L, self.norms)], -1)
        def forward(self, x, m):
            return self.fc(self.drop(self.feats(x, m)))
    net = Net().to(dev)
    net.norms.float(); net.fc.float()
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    trainable = [p for p in net.parameters() if p.requires_grad]
    log('trainable M', round(sum(p.numel() for p in trainable) / 1e6, 1))

    # length-bucketed batches, identical plan on every rank, then strided by rank
    rng = np.random.RandomState(CFG['seed'])
    def plan():
        order = rng.permutation(len(P_ids))
        chunks = [order[i:i + CFG['bs'] * 50] for i in range(0, len(order), CFG['bs'] * 50)]
        bl = []
        for c in chunks:
            c = sorted(c, key=lambda i: len(P_ids[i]))
            bl += [c[i:i + CFG['bs']] for i in range(0, len(c) - CFG['bs'] + 1, CFG['bs'])]
        rng.shuffle(bl); bl = bl[:len(bl) // world * world]
        return bl[rank::world]
    batches = [b for _ in range(CFG['epochs']) for b in plan()]
    def collate(li, ids):
        n = max(len(ids[i]) for i in li)
        x = np.full((len(li), n), PAD, np.int64); m = np.zeros((len(li), n), np.int64)
        for k, i in enumerate(li): x[k, :len(ids[i])] = ids[i]; m[k, :len(ids[i])] = 1
        return torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)

    cnt = np.bincount(P_y, minlength=len(LABELS)).astype(np.float64)
    CW = torch.tensor(cnt.mean() / np.maximum(cnt, 1), dtype=torch.float32, device=dev)
    lora_p = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith('base.')]
    head_p = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith('base.')]
    opt = torch.optim.AdamW([{'params': lora_p, 'lr': CFG['lr_a']}, {'params': head_p, 'lr': CFG['lr_h']}], weight_decay=CFG['wd'])
    total = len(batches); warm = max(1, int(CFG['warm'] * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.cuda.amp.GradScaler()
    os.makedirs(OUT, exist_ok=True)
    def save(tag):
        if rank: return
        from safetensors.torch import save_file
        sd = {n: p.detach().float().cpu().contiguous() for n, p in net.named_parameters() if p.requires_grad}
        sd.update({'buf.' + n: b.detach().float().cpu().contiguous() for n, b in net.named_buffers() if not n.startswith('base.')})
        save_file(sd, f'{OUT}/lora_head_{tag}.safetensors'); log('saved', tag)

    from sklearn.metrics import balanced_accuracy_score
    def official_eval(tag):
        net.eval(); torch.cuda.empty_cache(); o = sorted(range(len(T_ids)), key=lambda i: len(T_ids[i])); Z = np.zeros((len(T_ids), len(LABELS)), np.float32)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for s0 in range(0, len(o), 8):
                li = o[s0:s0 + 8]; x, m = collate(li, T_ids); Z[li] = net(x, m).float().cpu().numpy()
        net.train(); torch.cuda.empty_cache()
        log(f'[{tag}] OFFICIAL-ROWS BA', round(balanced_accuracy_score(T_y, Z.argmax(1)), 4), 'nonfinite rows', int((~np.isfinite(Z).all(1)).sum())); return Z
    snap = lambda: [p.detach().to('cpu', copy=True) for p in trainable]
    good = snap(); nskip = 0
    log('steps/rank', total)
    ddp.train(); tl = time.time(); toks = 0; run = []
    for step, li in enumerate(batches, 1):
        x, m = collate(li, P_ids); y = torch.from_numpy(P_y[li].astype(np.int64)).to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = ddp(x, m)
        loss = nn.functional.cross_entropy(logits.float(), y, weight=CW)
        opt.zero_grad(set_to_none=True)
        fin = torch.tensor([float(torch.isfinite(loss))], device=dev); dist.all_reduce(fin, op=dist.ReduceOp.MIN)
        if fin.item() < 1:
            nskip += 1; log(f'step {step}: non-finite loss, skipped (total {nskip})'); sch.step(); continue
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        toks += int(m.sum()) * world; run.append(float(loss))
        if step % 50 == 0:
            ok = torch.tensor([float(all(torch.isfinite(p).all() for p in trainable))], device=dev); dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if ok.item() < 1:
                with torch.no_grad():
                    for p, g in zip(trainable, good): p.copy_(g)
                opt.state.clear(); scaler = torch.cuda.amp.GradScaler(init_scale=1024); log(f'step {step}: non-finite weights, rolled back to last good snapshot')
            else: good = snap()
        if step % 25 == 0 or step <= 3:
            log(f'step {step}/{total} loss {np.mean(run[-25:]):.4f} scale {scaler.get_scale():.0f} tok/s {toks / (time.time() - tl):.0f} '
                f'eta {(time.time() - tl) / step * (total - step) / 60:.0f}m')
        if step % CFG['ckpt_every'] == 0:
            save('last')
            if rank == 0: official_eval(f'step {step}')
            dist.barrier()
    save('final')

    del opt; good = None; torch.cuda.empty_cache()
    if rank == 0:
        Z = official_eval('final')
        pred = Z.argmax(1); ba = balanced_accuracy_score(T_y, pred)
        rec = {l: round(float((pred[T_y == i] == i).mean()), 3) for i, l in enumerate(LABELS)}
        log('OFFICIAL-ROWS BA', round(ba, 4)); log('recall', json.dumps(rec))
        np.save(f'{OUT}/train_logits.npy', Z)
        json.dump({'cfg': CFG, 'labels': LABELS, 'ba': ba, 'recall': rec, 'losses': run, 'secs': time.time() - T0}, open(f'{OUT}/meta.json', 'w'))
        log('done')
    dist.barrier(); dist.destroy_process_group()

if __name__ == '__main__':
    import torch, torch.multiprocessing as mp
    world = torch.cuda.device_count(); print('gpus', world, flush=True)
    mp.spawn(main, args=(world,), nprocs=world, join=True)
