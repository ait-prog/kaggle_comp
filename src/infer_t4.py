import os, sys, json, glob, time
import numpy as np, pandas as pd, torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

def run(md, wpath, meta, texts, gpu, out):
    cfg = meta['cfg']; L = cfg['layers']; dev = torch.device(f'cuda:{gpu}')
    tok = AutoTokenizer.from_pretrained(md)
    eos = tok.eos_token_id; pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    mc = AutoConfig.from_pretrained(md); mc.num_hidden_layers = cfg['keep']; mc.use_cache = False
    if getattr(mc, 'layer_types', None) is not None: mc.layer_types = mc.layer_types[:cfg['keep']]
    base = AutoModel.from_pretrained(md, config=mc, dtype=torch.float16)
    sd = load_file(wpath)
    s = cfg['alpha'] / cfg['r']
    with torch.no_grad():
        for n, mod in base.named_modules():
            if n + '.A' in sd:
                a, b = sd[n + '.A'], sd[n + '.B']
                mod.weight += (s * (b @ a)).to(mod.weight.dtype)
    base.to(dev).eval()
    H = mc.hidden_size
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in L]); fc = nn.Linear(H * len(L), len(meta['labels']))
    norms.load_state_dict({k[len('head.norms.'):]: v for k, v in sd.items() if k.startswith('head.norms.')})
    fc.load_state_dict({k[len('head.fc.'):]: v for k, v in sd.items() if k.startswith('head.fc.')})
    norms.to(dev).float(); fc.to(dev).float()
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg['buckets'][-1] - 1)['input_ids']
    ids = [x + [eos] for x in ids]
    order = np.argsort([len(x) for x in ids], kind='stable')
    lg = np.zeros((len(ids), len(meta['labels'])), np.float32); bad = 0
    def fwd(li, dtype):
        n = max(len(ids[i]) for i in li)
        x = torch.full((len(li), n), pad, dtype=torch.long); m = torch.zeros((len(li), n), dtype=torch.long)
        for k, i in enumerate(li): x[k, :len(ids[i])] = torch.tensor(ids[i]); m[k, :len(ids[i])] = 1
        x, m = x.to(dev), m.to(dev)
        with torch.autocast('cuda', dtype=dtype):
            hs = base(input_ids=x, attention_mask=m, output_hidden_states=True, use_cache=False).hidden_states
        mm = m.unsqueeze(-1).float(); d = mm.sum(1)
        z = torch.cat([nm((hs[l].float() * mm).sum(1) / d) for l, nm in zip(L, norms)], -1)
        return fc(z)
    with torch.inference_mode():
        for s0 in range(0, len(order), 16):
            li = order[s0:s0 + 16].tolist()
            o = fwd(li, torch.float16)
            if not torch.isfinite(o).all():
                bad += 1; base.to(torch.bfloat16); o = fwd(li, torch.bfloat16); base.to(torch.float16)
            lg[li] = o.float().cpu().numpy()
    np.save(out, lg)
    print(json.dumps({'gpu': gpu, 'rows': len(ids), 'bad_batches': bad}), flush=True)

if __name__ == '__main__':
    a = json.loads(sys.argv[1])
    df = pd.read_parquet(a['input']) if a['input'].endswith('.parquet') else pd.read_csv(a['input'])
    part = np.array_split(np.arange(len(df)), a['n'])[a['gpu']]
    run(a['model_dir'], a['weights'], json.load(open(a['meta'])), df.text.fillna('').astype(str).values[part], 0, a['out'])
