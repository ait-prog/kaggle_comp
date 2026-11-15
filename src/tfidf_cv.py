"""TF-IDF LinearSVC on curated pool (KGR, cap) + official rows from 4 folds; OOF decision scores on the 5th fold.
usage: python src/tfidf_cv.py --cap 3000 --wofficial 3"""
import argparse, numpy as np, pandas as pd, time
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score as BA
ap = argparse.ArgumentParser(); ap.add_argument('--cap', type=int, default=3000); ap.add_argument('--wofficial', type=float, default=3.0)
ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
tr = pd.read_csv('data/train.csv'); L = sorted(tr.model.unique()); lid = {l: i for i, l in enumerate(L)}; y = tr.model.map(lid).values
fold = pd.read_csv('data/folds.csv').set_index('id').loc[tr.id, 'fold'].values
p = pd.read_parquet('ext/pool/ntl_open_pool.parquet', columns=['text', 'model', 'source_dataset', 'gen_route', 'gen_reasoning', 'flag_kb_proxy_after_ruling'])
m = ~p.flag_kb_proxy_after_ruling.values
v = (p.source_dataset.str.contains('vertex', case=False) | p.gen_route.str.contains('vertex', case=False)).values
m &= (p.model.values != 'gemini-3.7-flash') | v; m &= (p.model.values != 'kimi-k3') | (p.gen_reasoning.values == 'high')
p = p[m].reset_index(drop=True); rng = np.random.RandomState(a.seed)
idx = np.concatenate([rng.choice(np.flatnonzero((p.model == l).values), min(a.cap, int((p.model == l).sum())), replace=False) for l in L])
P_txt, P_y = p.text.values[idx], p.model.map(lid).values[idx]
D = np.zeros((len(tr), 18), np.float32)
for k in range(5):
    t = time.time(); tm = fold != k
    txt = np.concatenate([P_txt, tr.text.values[tm]]); yy = np.concatenate([P_y, y[tm]])
    sw = np.concatenate([np.ones(len(P_y)), np.full(tm.sum(), a.wofficial)])
    vc = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), sublinear_tf=True, min_df=3, max_features=300000, dtype=np.float32)
    vw = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000, token_pattern=r"(?u)\b\w+\b|[^\w\s]", dtype=np.float32)
    X = hstack([vc.fit_transform(txt), vw.fit_transform(txt)]).tocsr()
    Xt = hstack([vc.transform(tr.text.values[~tm]), vw.transform(tr.text.values[~tm])]).tocsr()
    D[~tm] = LinearSVC(C=0.3, class_weight='balanced').fit(X, yy, sample_weight=sw).decision_function(Xt)
    print(f'fold {k} BA {BA(y[~tm], D[~tm].argmax(1)):.4f} {time.time() - t:.0f}s', flush=True)
print('OOF BA', round(BA(y, D.argmax(1)), 4))
np.save(f'out/tfidf_cv_cap{a.cap}_w{a.wofficial:g}_s{a.seed}.npy', D)
