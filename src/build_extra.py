"""Build ext/extra_clean.parquet: public NTL passages not already in the open pool, cleaned with the pool's own rules.

Drops: exact duplicates of pool/official rows; rows the pool author removed (tagged in riccapucho's superset);
Benchmarks-proxy sources (host ruling); rows with TF-IDF cosine > 0.30 to any official train passage (pool's cap);
gemini-3.7 rows not from Vertex; kimi-k3 rows without known reasoning=high; glm rows flagged low reasoning.
"""
import re, hashlib, glob
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

norm = lambda s: re.sub(r'\s+', ' ', str(s)).strip().lower()
h = lambda s: hashlib.md5(norm(s).encode()).hexdigest()
D = 'ext/pubdata/'
pool = pd.read_parquet('ext/pool/ntl_open_pool.parquet', columns=['text']); PH = set(pool.text.map(h))
tr = pd.read_csv('data/train.csv'); TH = set(tr.text.map(h)); LAB = set(tr.model)

rc = pd.read_parquet(D + 'riccapucho__gaggle-ntl-superset-v1/gaggle_ntl_superset.parquet')
rc['h'] = rc.text.map(h)
BAD = set(rc.h[(rc.quality_flag != '') | (rc.label_risk != '') | (rc.host_format_flag != '')])

frames = []
def add(name, d, orig=None, reasoning=None):
    d = d[d.model.isin(LAB)].copy()
    d['src'] = name
    d['orig'] = (d[orig].astype(str) if orig else name)
    d['reasoning'] = d[reasoning].astype(str).str.lower() if reasoning else ''
    frames.append(d[['text', 'model', 'src', 'orig', 'reasoning']])

add('riccapucho', rc[~rc.h.isin(BAD)], 'original_dataset')
g = pd.read_parquet(D + 'gtuli12__ntl-own-pool/pool_own_v3.parquet')
g = g[~g.kimi_reasoning_off.fillna(False).astype(bool) & ~g.glm_low_reasoning.fillna(False).astype(bool)]
add('gtuli12', g, 'source')
j = pd.read_parquet(D + 'jeevajoji__name-that-llm-extra-passages-226k/ntl_extra_merged.parquet')
add('jeevajoji', j, 'source', 'reasoning')
add('illiadolenko', pd.read_csv(D + 'illiadolenko__ntl-merged-corpus-v2/merged_corpus.csv'), 'source')
add('mathleteutd', pd.read_parquet(D + 'mathleteutd__ntl-public-pool-v2/ntl_public_pool_v2.parquet'), 'src')
add('lourenswalters', pd.read_csv(D + 'lourenswalters__ntl-community-vetted-05/community_vetted_05.csv'), 'source')
add('prvsiyan_v1', pd.read_parquet(D + 'prvsiyan__ntl-public-in-style-pool-v1/pool_v1.parquet'), 'source')
add('samarth1006', pd.read_parquet(D + 'samarth1006__nthl-pool-v10/pool_v10.parquet'), 'src')
add('gougou1', pd.read_csv(D + 'gougou1__ntllm-709-extra/extra_passages_709.csv'))
add('nischaydnk', pd.read_parquet(D + 'nischaydnk__ntl-topic-synthetic-18llms/train_usable.parquet'))
s = pd.read_csv(D + 'shakshamkarkiii__ntl-own-matched-b1/own_b1_clean.csv')
add('shakshamkarkiii', s[s.quarantine.fillna(False).astype(str).str.lower().isin(['false', '0', ''])] if 'quarantine' in s else s)
# excluded: nimithashreea, dhananjayk2609, tamerlanomralinov, aitahmednasser (Benchmarks proxy / Kaggle AI gen)

x = pd.concat(frames, ignore_index=True)
x['h'] = x.text.map(h)
n0 = len(x)
x = x[~x.h.isin(PH) & ~x.h.isin(TH) & ~x.h.isin(BAD)].drop_duplicates('h')
log = {'raw': n0, 'new_unique': len(x)}
ban = re.compile(r'bench|kaggle.?ai.?gen|kgen|model.?proxy|kaggle.?proxy|kb_?proxy', re.I)
x = x[~x.orig.str.contains(ban) & ~x.src.str.contains(ban)]; log['no_benchmarks'] = len(x)
x = x[(x.model != 'gemini-3.7-flash') | x.orig.str.contains('vertex', case=False)]; log['gemini37_vertex'] = len(x)
x = x[(x.model != 'kimi-k3') | x.reasoning.str.contains('high')]; log['kimi_high'] = len(x)
nw = x.text.str.split().str.len(); x = x[(nw >= 30) & (nw <= 300) & ~x.text.str.contains('\n\n|^#|\\*\\*', regex=True)]; log['format'] = len(x)

v = TfidfVectorizer(stop_words='english', sublinear_tf=True, min_df=2, dtype=np.float32).fit(tr.text)
T = v.transform(tr.text).T.tocsc(); mx = np.zeros(len(x), np.float32); X = v.transform(x.text)
for s0 in range(0, len(x), 5000):
    mx[s0:s0 + 5000] = (X[s0:s0 + 5000] @ T).max(1).toarray().ravel()
x = x[mx <= 0.30]; log['not_near_train'] = len(x)
x = x.drop(columns='h').reset_index(drop=True)
x.to_parquet('ext/extra_clean.parquet')
print(log); print(x.model.value_counts().to_dict()); print(x.src.value_counts().to_dict())
