import sys,time,csv,math,statistics as st
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim'))
from functools import lru_cache
from multiprocessing import Pool
from graph_layouts import make_ladder
from engine import Sim,VARIANTS,enumerate_valid_walks,Weights
import runner
NP,NG,K=24,20,100
@lru_cache(maxsize=None)
def lay(i): return make_ladder(rows=20,cols=20,n_ghosts=NG,n_priorities=NP,seed=0)[i]
@lru_cache(maxsize=None)
def walks(i):
    l=lay(i); return enumerate_valid_walks(l.G,l.priorities,0,K,t_seconds=400,pos=l.pos,v_g=4.0)
def job(a):
    i,var,sd,hor=a
    s=Sim(lay(i),8,4,VARIANTS[var],seed=sd,weights=runner.make_weights(400),valid_walks=walks(i)); log=s.run(hor); n=len(log)
    return a,(sum(r["mean_idleness"] for r in log)/n,sum(r["ghost_mean_idleness"] for r in log)/n,max(r["ghost_max_idleness"] for r in log))
if __name__=="__main__":
    t=time.time(); l=lay(0); print("nodes",l.G.number_of_nodes(),"edges",l.G.number_of_edges(),"prio",len(l.priorities),"ghosts",len(l.ghosts)); l2=lay(1); print("thin edges",l2.G.number_of_edges())
    w=walks(0); print("walks",len(w),"build s",round(time.time()-t,1))
    cov=set(n for x in w for n in x); print("uncovered nodes M-full",l.G.number_of_nodes()-len(cov))
    w=walks(1); cov=set(n for x in w for n in x); print("walks thin",len(w),"uncovered",l2.G.number_of_nodes()-len(cov))
    t=time.time(); job((0,"full",0,600)); print("600 s sim takes",round(time.time()-t,1),"s")
