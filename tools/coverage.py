import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim'))
import statistics as st
from functools import lru_cache
from multiprocessing import Pool
import runner
from runner import _layout, make_weights
from engine import Sim,VARIANTS,enumerate_valid_walks,Weights
@lru_cache(maxsize=None)
def W(T,K):
    l=_layout(1); return enumerate_valid_walks(l.G,l.priorities,0,K,t_seconds=T,pos=l.pos,v_g=4.0)
def job(a):
    T,K,fleet,sd=a
    l=_layout(1); walks=W(T,K); nodes=list(l.G.nodes())
    s=Sim(l,*fleet,VARIANTS["full"],seed=sd,weights=make_weights(T),valid_walks=walks)
    vis={n:0 for n in nodes}; ov=s.visit
    def visit(n,*x,**k):
        if n in vis: vis[n]+=1
        return ov(n,*x,**k)
    s.visit=visit; pk=[0]; o=s.step; c=[0]
    def step():
        o(); c[0]+=1
        if c[0]%int(10/s.dt)==0: pk[0]=max(pk[0],max(s.idleness(n) for n in nodes))
    s.step=step; log=s.run(7200); n=len(log)
    gm=sum(r["mean_idleness"] for r in log)/n; am=sum(r["ghost_mean_idleness"] for r in log)/n
    return a,(sum(v==0 for v in vis.values()),sum(v<=2 for v in vis.values()),pk[0],gm,gm+am)
if __name__=="__main__":
    configs=[(400,500),(400,None),(600,500),(800,500),(1200,500)]
    for T,K in configs: W(T,K)
    cov={}
    for T,K in configs:
        w=W(T,K); l=_layout(1); c=set(n for x in w for n in x)
        starts={}
        for x in w:
            for n in set(x): starts.setdefault(n,set()).add(x[0])
        cov[(T,K)]=(len(w),100-len(c),sum(1 for n in l.G.nodes() if len(starts.get(n,()))<=2))
    jobs=[(T,K,f,s) for T,K in configs for f in ((4,2),(6,3)) for s in range(100,105)]
    with Pool() as p: res=p.map(job,jobs)
    agg={}
    for (T,K,f,s),x in res: agg.setdefault((T,K,f),[]).append(x)
    print("T    K     walks  nodes-on-no-walk  nodes-on-walks-from<=2-starts")
    for (T,K),c in cov.items(): print(f"{T:4} {str(K):5} {c[0]:6} {c[1]:10} {c[2]:20}")
    print("\nM2, full variant, 5 seeds: never-visited nodes | visited<=2 times | ground peak | ground mean | combined mean")
    for (T,K) in configs:
        for f in ((4,2),(6,3)):
            L=agg[(T,K,f)]; print(f"T={T:4} K={str(K):5} fleet {f}: "+" | ".join(f"{st.mean(x[i] for x in L):7.1f}" for i in range(5)))
