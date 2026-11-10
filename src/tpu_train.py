import os, sys, json, time, math, glob, functools
import numpy as np, pandas as pd

CFG = dict(
    run='dev2', smoke=True, model='8b-base', keep=24, layers=[3, 6, 9, 12, 16, 20, 24],
    r=16, alpha=32, ldrop=0.05, cap=12000, kimi_high=True, epochs=1, gbs=64,
    buckets=[128, 256, 384], lr_a=1e-4, lr_h=1e-3, wd=0.01, warm=0.05, fsdp=False,
    smoke_steps=12, seed=0,
)
if len(sys.argv) > 1: CFG.update(json.loads(sys.argv[1]))
print(json.dumps(CFG), flush=True)
T0 = time.time()
def log(*a):
    print(f'[{time.time() - T0:7.0f}s]', *a, flush=True)

import torch
from torch import nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.utils.checkpoint import checkpoint as xckpt
import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
log('torch', torch.__version__, 'xla', torch_xla.__version__, 'tf', transformers.__version__, 'cpus', os.cpu_count())
try:
    import psutil; log('ram GB', round(psutil.virtual_memory().total / 2**30))
except Exception: pass

xr.use_spmd()
N = xr.global_runtime_device_count()
mesh = xs.Mesh(np.arange(N), (N, 1), ('fsdp', 'model'))
dev = xm.xla_device()
log('devices', N)

def find_model(tag):
    want = {'8b-base': (4096, 36), '14b-base': (5120, 40), '4b-base': (2560, 36), '1.7b-base': (2048, 28)}[tag]
    for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try: j = json.load(open(c))
        except Exception: continue
        if j.get('model_type') == 'qwen3' and (j.get('hidden_size'), j.get('num_hidden_layers')) == want and tag.split('-')[0] in c.lower():
            return os.path.dirname(c)
    raise SystemExit('model not found ' + tag)
MD = find_model(CFG['model'])
log('model dir', MD)

comp = next(p for p in glob.glob('/kaggle/input/**/train.csv', recursive=True) if 'name-that-llm' in p)
tr = pd.read_csv(comp)
pool_path = next(iter(glob.glob('/kaggle/input/**/ntl_open_pool.parquet', recursive=True)))
pool = pd.read_parquet(pool_path, columns=['text', 'model', 'gen_reasoning'])
LABELS = sorted(tr.model.unique())
lid = {l: i for i, l in enumerate(LABELS)}
if CFG['kimi_high']:
    pool = pool[(pool.model != 'kimi-k3') | (pool.gen_reasoning == 'high')]
log('pool', len(pool), 'train', len(tr))

tok = AutoTokenizer.from_pretrained(MD)
EOS = tok.eos_token_id
PAD = tok.pad_token_id if tok.pad_token_id is not None else EOS
def enc(texts):
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=CFG['buckets'][-1] - 1)['input_ids']
    return [x + [EOS] for x in ids]
t = time.time()
P_ids = enc(pool.text.values); P_y = pool.model.map(lid).values
T_ids = enc(tr.text.values); T_y = tr.model.map(lid).values
log('tokenized', f'{time.time() - t:.0f}s', 'pool len p50/p99', int(np.median([len(x) for x in P_ids])), int(np.percentile([len(x) for x in P_ids], 99)))

def bucket_of(n):
    for b in CFG['buckets']:
        if n <= b: return b
    return CFG['buckets'][-1]

def make_batches(idx, ids, rng, gbs):
    by = {}
    for i in idx: by.setdefault(bucket_of(len(ids[i])), []).append(i)
    out = []
    for b, li in by.items():
        li = np.array(li); rng.shuffle(li)
        bs = gbs * max(1, CFG['buckets'][-1] // b) // 1
        bs = min(bs, gbs * 2)
        for s in range(0, len(li) - bs + 1, bs): out.append((b, li[s:s + bs]))
    rng.shuffle(out)
    return out

def collate(b, li, ids, ys=None):
    x = np.full((len(li), b), PAD, np.int64); m = np.zeros((len(li), b), np.int64)
    for k, i in enumerate(li):
        s = ids[i][:b]; x[k, :len(s)] = s; m[k, :len(s)] = 1
    out = [torch.from_numpy(x), torch.from_numpy(m)]
    if ys is not None: out.append(torch.from_numpy(ys[li].astype(np.int64)))
    return out

class LoRA(nn.Module):
    def __init__(self, base, r, alpha, p):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.s = alpha / r; self.drop = nn.Dropout(p)
    def forward(self, x):
        return self.base(x) + ((self.drop(x).float() @ self.A.t()) @ self.B.t() * self.s).to(x.dtype)

mc = AutoConfig.from_pretrained(MD)
mc.num_hidden_layers = CFG['keep']; mc.use_cache = False
if getattr(mc, 'layer_types', None) is not None: mc.layer_types = mc.layer_types[:CFG['keep']]
t = time.time()
base = AutoModel.from_pretrained(MD, config=mc, dtype=torch.bfloat16)
log('loaded', f'{time.time() - t:.0f}s', 'params', sum(p.numel() for p in base.parameters()) / 1e9)
for p in base.parameters(): p.requires_grad_(False)
TGT = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
for layer in base.layers:
    for parent in (layer.self_attn, layer.mlp):
        for n in TGT:
            if hasattr(parent, n): setattr(parent, n, LoRA(getattr(parent, n), CFG['r'], CFG['alpha'], CFG['ldrop']))
H = mc.hidden_size; L = CFG['layers']
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in L])
        self.fc = nn.Linear(H * len(L), len(LABELS)); self.drop = nn.Dropout(0.1)
    def feats(self, hs, m):
        m = m.unsqueeze(-1).float(); d = m.sum(1)
        return torch.cat([nm((hs[l].float() * m).sum(1) / d) for l, nm in zip(L, self.norms)], -1)
head = Head()

for layer in base.layers:
    f = layer.forward
    def wrapped(*a, _f=f, **k):
        if not torch.is_grad_enabled(): return _f(*a, **k)
        return xckpt(functools.partial(_f, **k), *a, use_reentrant=True)
    layer.forward = wrapped

base.to(dev); head.to(dev)
if CFG['fsdp']:
    for n_, p in base.named_parameters():
        if p.dim() == 2 and p.shape[0] % N == 0 and not n_.endswith('.A') and not n_.endswith('.B'):
            xs.mark_sharding(p, mesh, ('fsdp', None))
for n_, p in base.named_parameters():
    if n_.endswith('.A') or n_.endswith('.B'): p.requires_grad_(True)
trainable = [p for n_, p in base.named_parameters() if p.requires_grad]
log('lora params M', sum(p.numel() for p in trainable) / 1e6)
emb = base.get_input_embeddings()
emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True) if torch.is_grad_enabled() else o)

def forward(x, m):
    xs.mark_sharding(x, mesh, ('fsdp', None)); xs.mark_sharding(m, mesh, ('fsdp', None))
    hs = base(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False).hidden_states
    assert len(hs) == CFG['keep'] + 1, len(hs)
    return head.feats(hs, m)

rng = np.random.RandomState(CFG['seed'])
def epoch_idx():
    li = []
    for c in range(len(LABELS)):
        w = np.where(P_y == c)[0]
        li.append(rng.choice(w, min(len(w), CFG['cap']), replace=False))
    return np.concatenate(li)
plan = [make_batches(epoch_idx(), P_ids, rng, CFG['gbs']) for _ in range(CFG['epochs'])]
if CFG['smoke']:
    sm = []
    for b in CFG['buckets']:
        sm += [x for x in plan[0] if x[0] == b][:CFG['smoke_steps']]
    plan = [sm]
cnt = np.array([min((P_y == c).sum(), CFG['cap']) for c in range(len(LABELS))], np.float64)
CW = torch.tensor((cnt.mean() / cnt) ** CFG.get('cw_pow', 1.0), dtype=torch.float32).to(dev)
log('class weights', np.round((cnt.mean() / cnt) ** CFG.get('cw_pow', 1.0), 2).tolist())
total = sum(len(p) for p in plan); warm = max(1, int(CFG['warm'] * total))
opt = torch.optim.AdamW([{'params': trainable, 'lr': CFG['lr_a']}, {'params': list(head.parameters()), 'lr': CFG['lr_h']}], weight_decay=CFG['wd'])
sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
log('steps', total)

def train_step(x, m, y):
    opt.zero_grad(set_to_none=True)
    z = forward(x, m)
    loss = nn.functional.cross_entropy(head.fc(head.drop(z)), y, weight=CW)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable + list(head.parameters()), 1.0)
    opt.step(); sch.step()
    return loss

base.train(); head.train()
step = 0; losses = []; tl = time.time(); seen = 0
for ep, batches in enumerate(plan):
    for b, li in batches:
        x, m, y = [v.to(dev) for v in collate(b, li, P_ids, P_y)]
        loss = train_step(x, m, y)
        xm.mark_step()
        step += 1; seen += len(li) * b
        if step % 25 == 0 or CFG['smoke']:
            lv = float(loss.detach().cpu()); losses.append(lv)
            log(f'step {step}/{total} b{b} bs{len(li)} loss {lv:.4f} tok/s {seen / (time.time() - tl):.0f}')
            if CFG['smoke']:
                try: log('mem', xm.get_memory_info(dev))
                except Exception as e: log('mem n/a', repr(e)[:100])

base.eval(); head.eval()
def infer(ids, idx):
    Z = np.zeros((len(idx), H * len(L)), np.float32)
    order = sorted(range(len(idx)), key=lambda k: len(ids[idx[k]]))
    bs = CFG['gbs']
    with torch.no_grad():
        for s in range(0, len(order), bs):
            pos = order[s:s + bs]; li = [idx[k] for k in pos]
            b = bucket_of(max(len(ids[i]) for i in li))
            pad = bs - len(li)
            li2 = li + [li[0]] * pad
            x, m = [v.to(dev) for v in collate(b, li2, ids)]
            z = forward(x, m); xm.mark_step()
            Z[pos] = z.cpu().numpy()[:len(li)]
    return Z
ev = np.arange(len(T_ids)) if not CFG['smoke'] else np.arange(256)
t = time.time()
Z = infer(T_ids, list(ev))
W = head.fc.weight.detach().cpu().numpy(); Bv = head.fc.bias.detach().cpu().numpy()
lg = Z @ W.T + Bv
from sklearn.metrics import balanced_accuracy_score
ba = balanced_accuracy_score(T_y[ev], lg.argmax(1))
log('eval train', len(ev), 'BA', round(ba, 4), 'acc', round(float((lg.argmax(1) == T_y[ev]).mean()), 4), f'{time.time() - t:.0f}s')
os.makedirs('/kaggle/working/out', exist_ok=True)
np.save('/kaggle/working/out/train_logits.npy', lg.astype(np.float32))
np.save('/kaggle/working/out/train_feats.npy', Z.astype(np.float16))
sd = {n_: p.detach().cpu() for n_, p in base.named_parameters() if p.requires_grad}
sd.update({'head.' + k: v.detach().cpu() for k, v in head.state_dict().items()})
from safetensors.torch import save_file
save_file({k: v.contiguous().float() for k, v in sd.items()}, '/kaggle/working/out/lora_head.safetensors')
json.dump({'cfg': CFG, 'labels': LABELS, 'ba': ba, 'losses': losses, 'secs': time.time() - T0}, open('/kaggle/working/out/meta.json', 'w'))
log('done')
