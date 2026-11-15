"""Fast CPU yardstick for pool-curation arms.

Train a char+word TF-IDF linear model on a per-label-capped pool sample, score balanced accuracy on all
2,699 official train rows (never used for fitting), and report paired deltas vs the baseline arm with a
cluster bootstrap over the 150 prompts.

usage: python src/harness.py [--cap 2500] [--seeds 0 1] [--arms B K KG KGR]
"""
import argparse, json, time
import numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument('--cap', type=int, default=2500)
ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
ap.add_argument('--arms', nargs='+', default=['B', 'K', 'KG', 'KGR'])
ap.add_argument('--out', default='out/harness')
a = ap.parse_args()

tr = pd.read_csv('data/train.csv')
pool = pd.read_parquet('ext/pool/ntl_open_pool.parquet', columns=[
    'text', 'model', 'source_dataset', 'gen_route', 'gen_reasoning', 'flag_kb_proxy_after_ruling'])
pool['is_extra'] = False
_x = pd.read_parquet('ext/extra_clean.parquet')
_x = pd.DataFrame({'text': _x.text, 'model': _x.model, 'source_dataset': _x.orig, 'gen_route': '',
                   'gen_reasoning': np.where(_x.model == 'kimi-k3', 'high', 'extra'), 'flag_kb_proxy_after_ruling': False, 'is_extra': True})
pool = pd.concat([pool, _x], ignore_index=True)
LABELS = sorted(tr.model.unique()); lid = {l: i for i, l in enumerate(LABELS)}
y_tr = tr.model.map(lid).values

def arm_mask(arm):
    m = np.ones(len(pool), bool) if 'E' in arm else ~pool.is_extra.values  # E: add cleaned public extra rows
    if 'K' in arm:   # host ruling: Benchmarks proxy quota is not for training data
        m &= ~pool.flag_kb_proxy_after_ruling.values
    if 'G' in arm:   # gemini-3.7: keep only Vertex-sourced rows (API routes 3.7 -> 3.8 elsewhere)
        vert = pool.source_dataset.str.contains('vertex', case=False) | pool.gen_route.str.contains('vertex', case=False)
        m &= (pool.model.values != 'gemini-3.7-flash') | vert.values
    if 'R' in arm:   # kimi-k3: keep only reasoning=high rows
        m &= (pool.model.values != 'kimi-k3') | (pool.gen_reasoning.values == 'high')
    if 'O' in arm:   # claude-opus-4-8: drop rows with unknown reasoning config (fidelity 0.45 vs 0.62-0.76)
        m &= (pool.model.values != 'claude-opus-4-8') | (pool.gen_reasoning.values != 'unknown')
    if 'Z' in arm:   # glm-5.3: keep default/minimal reasoning only (fidelity 0.90/0.81 vs low 0.68)
        m &= (pool.model.values != 'glm-5.3') | np.isin(pool.gen_reasoning.values, ['default', 'minimal'])
    return m

def sample(mask, seed):
    rng = np.random.RandomState(seed); idx = []
    sub = pool[mask]
    for l in LABELS:
        w = np.flatnonzero((sub.model == l).values)
        idx.append(sub.index.values[rng.choice(w, min(len(w), a.cap), replace=False)])
    return np.concatenate(idx)

# cluster bootstrap over prompts, shared across arms so deltas are paired
groups = tr.prompt_id.values; ug = np.unique(groups)
rng = np.random.RandomState(123)
BOOT = [np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(ug, len(ug))]) for _ in range(1000)]

def ba(pred, rows=None):
    if rows is None: return balanced_accuracy_score(y_tr, pred)
    return balanced_accuracy_score(y_tr[rows], pred[rows])

res, preds = [], {}
for seed in a.seeds:
    for arm in a.arms:
        t = time.time()
        idx = sample(arm_mask(arm), seed)
        X_txt = pool.text.values[idx]; y = pool.model.map(lid).values[idx]
        vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=3, max_features=300000, dtype=np.float32)
        vw = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000, token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32)
        X = hstack([vc.fit_transform(X_txt), vw.fit_transform(X_txt)]).tocsr()
        Xt = hstack([vc.transform(tr.text.values), vw.transform(tr.text.values)]).tocsr()
        clf = LinearSVC(C=0.3, class_weight='balanced').fit(X, y)
        dec = clf.decision_function(Xt); p = dec.argmax(1)
        preds[(arm, seed)] = p
        np.save(f'{a.out}_{arm}_s{seed}.npy', dec.astype(np.float32))
        r = dict(arm=arm, seed=seed, n=len(idx), ba=round(ba(p), 4), secs=round(time.time() - t))
        res.append(r); print(json.dumps(r), flush=True)

base = a.arms[0]
print('\npaired deltas vs', base, '(mean over seeds; 95% prompt-cluster bootstrap CI)')
for arm in a.arms[1:]:
    d = np.array([[ba(preds[(arm, s)], b) - ba(preds[(base, s)], b) for s in a.seeds] for b in BOOT]).mean(1)
    pt = np.mean([ba(preds[(arm, s)]) - ba(preds[(base, s)]) for s in a.seeds])
    print(f'{arm:6s} {pt:+.4f}  [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]')
per = {arm: np.mean([[(preds[(arm, s)][y_tr == c] == c).mean() for c in range(18)] for s in a.seeds], 0) for arm in a.arms}
print('\nper-class recall'); print(pd.DataFrame(per, index=LABELS).round(3).to_string())
json.dump(res, open(f'{a.out}_results.json', 'w'), indent=1)
