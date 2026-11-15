# dev8: start from the dev6 adapter+head (pool-pretrained) and fine-tune on official rows. 5 prompt-grouped folds -> OOF logits,
# plus a full-fit adapter for submission. Each GPU runs its own jobs independently (no DDP): rank0 folds 0,2,4; rank1 folds 1,3,full.
# Curation (CPU harness, +0.030 BA [+0.016,+0.044] vs raw): drop Benchmarks-proxy rows after the host ruling,
# gemini-3.7 from Vertex sources only, kimi-k3 reasoning=high only. Eval: all 2,699 official train rows (never trained on).
import os, sys, json, time, math, glob
import numpy as np, pandas as pd

CFG = dict(run='dev8', ft_epochs=3, ft_lr_a=5e-5, ft_lr_h=3e-4, ft_bs=16, hidden=2560, n_layers=36, keep=24, layers=[4, 8, 12, 16, 20, 24], r=16, alpha=32, ldrop=0.05,
           cap=3000, epochs=1, bs=16, maxlen=320, lr_a=2e-4, lr_h=1e-3, wd=0.01, warm=0.05, seed=0, ckpt_every=400)
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
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29511')
    torch.cuda.set_device(rank); dev = torch.device('cuda', rank)
    T0 = time.time()
    def log(*a):
        print(f'[r{rank}]', f'[{time.time() - T0:6.0f}s]', *a, flush=True)
    torch.manual_seed(CFG['seed'])
    tr, pool, LABELS = load_data(); lid = {l: i for i, l in enumerate(LABELS)}
    log('cfg', json.dumps(CFG))
    MD = find_model(); log('model', MD)
    tok = AutoTokenizer.from_pretrained(MD); EOS = tok.eos_token_id
    PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
    enc = lambda t: [x + [EOS] for x in tok(list(t), add_special_tokens=False, truncation=True, max_length=CFG['maxlen'] - 1)['input_ids']]
    T_ids, T_y = enc(tr.text.values), tr.model.map(lid).values

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
    base = AutoModel.from_pretrained(MD, config=mc, torch_dtype=torch.float16, attn_implementation='sdpa').to(dev)
    for p in base.parameters(): p.requires_grad_(False)
    for layer in base.layers:
        for parent in (layer.self_attn, layer.mlp):
            for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
                if hasattr(parent, n): setattr(parent, n, LoRA(getattr(parent, n), CFG['r'], CFG['alpha'], CFG['ldrop']))
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
    trainable = [p for p in net.parameters() if p.requires_grad]
    log('trainable M', round(sum(p.numel() for p in trainable) / 1e6, 1))


    from safetensors.torch import load_file
    ck = next(iter(glob.glob('/kaggle/input/**/lora_head_final.safetensors', recursive=True))); log('init from', ck)
    SD0 = load_file(ck)
    def reset():
        pn = dict(net.named_parameters()); bn = dict(net.named_buffers())
        with torch.no_grad():
            for k, v in SD0.items():
                if k.startswith('buf.'): bn[k[4:]].copy_(v)
                else: pn[k].copy_(v.to(pn[k].dtype))
    U = sorted(tr.prompt_id.unique()); FM = {p: i % 5 for i, p in enumerate(U)}; FOLD = tr.prompt_id.map(FM).values
    def collate(li, ids):
        n = max(len(ids[i]) for i in li)
        x = np.full((len(li), n), PAD, np.int64); m = np.zeros((len(li), n), np.int64)
        for k, i in enumerate(li): x[k, :len(ids[i])] = ids[i]; m[k, :len(ids[i])] = 1
        return torch.from_numpy(x).to(dev), torch.from_numpy(m).to(dev)
    def predict(rows):
        net.eval(); Z = np.zeros((len(rows), len(LABELS)), np.float32); o = sorted(range(len(rows)), key=lambda i: len(T_ids[rows[i]]))
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            for s in range(0, len(o), 32):
                li = o[s:s + 32]; x, m = collate([rows[i] for i in li], T_ids); Z[li] = net(x, m).float().cpu().numpy()
        return Z
    from sklearn.metrics import balanced_accuracy_score
    os.makedirs(OUT, exist_ok=True)
    jobs = [[0, 2, 4], [1, 3, 'full']][rank] if world > 1 else [0, 1, 2, 3, 4, 'full']
    for job in jobs:
        reset(); t0 = time.time()
        tr_rows = np.flatnonzero(FOLD != job) if job != 'full' else np.arange(len(T_ids))
        te_rows = np.flatnonzero(FOLD == job) if job != 'full' else None
        if te_rows is not None:
            Z0 = predict(te_rows); log(f'fold {job} before ft BA', round(balanced_accuracy_score(T_y[te_rows], Z0.argmax(1)), 4))
        lora_p = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith('base.')]
        head_p = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith('base.')]
        opt = torch.optim.AdamW([{'params': lora_p, 'lr': CFG['ft_lr_a']}, {'params': head_p, 'lr': CFG['ft_lr_h']}], weight_decay=CFG['wd'])
        rng = np.random.RandomState(CFG['seed'] + (job if job != 'full' else 9)); bs = CFG['ft_bs']
        batches = []
        for _ in range(CFG['ft_epochs']):
            o = rng.permutation(tr_rows); batches += [o[i:i + bs] for i in range(0, len(o) - bs + 1, bs)]
        total = len(batches); warm = max(1, int(0.1 * total))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        scaler = torch.cuda.amp.GradScaler(); run = []
        for step, li in enumerate(batches, 1):
            net.train(); x, m = collate(li, T_ids); y = torch.from_numpy(T_y[li].astype(np.int64)).to(dev)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = net(x, m)
            loss = nn.functional.cross_entropy(logits.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); scaler.step(opt); scaler.update(); sch.step(); run.append(float(loss))
            if step % 40 == 0: log(f'job {job} step {step}/{total} loss {np.mean(run[-40:]):.4f}')
            if te_rows is not None and step % (total // CFG['ft_epochs']) == 0:
                Ze = predict(te_rows); log(f'fold {job} after epoch {step // (total // CFG["ft_epochs"])} BA', round(balanced_accuracy_score(T_y[te_rows], Ze.argmax(1)), 4))
        if te_rows is not None:
            Z1 = predict(te_rows); np.save(f'{OUT}/oof_fold{job}.npy', Z1); np.save(f'{OUT}/oof_fold{job}_before.npy', Z0)
            log(f'fold {job} FINAL BA', round(balanced_accuracy_score(T_y[te_rows], Z1.argmax(1)), 4), 'secs', round(time.time() - t0))
        else:
            from safetensors.torch import save_file
            sd = {n: p.detach().float().cpu().contiguous() for n, p in net.named_parameters() if p.requires_grad}
            sd.update({'buf.' + n: b.detach().float().cpu().contiguous() for n, b in net.named_buffers() if not n.startswith('base.')})
            save_file(sd, f'{OUT}/lora_head_fullft.safetensors'); log('saved full-fit', round(time.time() - t0))
    log('done')

if __name__ == '__main__':
    import torch, torch.multiprocessing as mp
    world = torch.cuda.device_count(); print('gpus', world, flush=True)
    mp.spawn(main, args=(world,), nprocs=world, join=True)
    # assemble OOF
    import pandas as pd, numpy as np
    from sklearn.metrics import balanced_accuracy_score
    tr = pd.read_csv(next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p))
    L = sorted(tr.model.unique()); y = tr.model.map({l: i for i, l in enumerate(L)}).values
    U = sorted(tr.prompt_id.unique()); FOLD = tr.prompt_id.map({p: i % 5 for i, p in enumerate(U)}).values
    Z = np.zeros((len(tr), 18), np.float32); Zb = np.zeros_like(Z)
    for k in range(5):
        r = np.flatnonzero(FOLD == k); Z[r] = np.load(f'{OUT}/oof_fold{k}.npy'); Zb[r] = np.load(f'{OUT}/oof_fold{k}_before.npy')
    np.save(f'{OUT}/oof_logits.npy', Z)
    pred = Z.argmax(1); print('OOF BA before ft', round(balanced_accuracy_score(y, Zb.argmax(1)), 4), 'after ft', round(balanced_accuracy_score(y, pred), 4), flush=True)
    print('recall', json.dumps({l: round(float((pred[y == i] == i).mean()), 3) for i, l in enumerate(L)}), flush=True)
