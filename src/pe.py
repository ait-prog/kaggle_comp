import sys,time,numpy as np,pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
R='D:/hack/Name That LLM/'
tr=pd.read_csv(R+'data/train.csv')
p=pd.read_parquet(R+'ext/openpool/ntl_open_pool.parquet',columns=['text','model','gen_reasoning','style_score','match_weight','flag_kb_proxy','prompt_canon'])
def sub(name):
    q=p
    if 'kimihigh' in name: q=q[(q.model!='kimi-k3')|(q.gen_reasoning=='high')]
    if 'cap' in name:
        c=int(name.split('cap')[1].split('_')[0])
        q=q.sample(frac=1,random_state=0).groupby('model').head(c)
    return q
def run(name):
    t=time.time();q=sub(name)
    vs=[TfidfVectorizer(sublinear_tf=True,lowercase=False,min_df=3,analyzer='char',ngram_range=(1,4),max_features=400000),
        TfidfVectorizer(sublinear_tf=True,lowercase=False,min_df=3,ngram_range=(1,2),max_features=300000,token_pattern=r'(?u)\b\w+\b|[^\w\s]')]
    X=hstack([v.fit_transform(q.text) for v in vs]).tocsr()
    Xt=hstack([v.transform(tr.text) for v in vs]).tocsr()
    m=LinearSVC(C=0.3,class_weight='balanced').fit(X,q.model)
    pr=m.predict(Xt)
    ba=balanced_accuracy_score(tr.model,pr)
    rec=pd.Series(pr==tr.model.values).groupby(tr.model.values).mean().round(2)
    print(name,len(q),round(ba,4),f'{time.time()-t:.0f}s',flush=True)
    print(rec.to_dict(),flush=True)
for n in sys.argv[1:]: run(n)
