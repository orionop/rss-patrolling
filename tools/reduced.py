import sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim'))
import statistics as st, math, csv, collections
from multiprocessing import Pool
import runner
from runner import _layout,_walks,make_weights
from engine import Sim,VARIANTS
FLEETS=[(4,1),(5,1),(4,2),(5,2)]
VS=("full","no_ghost_credit","fixed_launch_nodes","nearest_ugv_recovery","tier0_ugv_only")
def job(a):
    mi,f,v,sd=a
    s=Sim(_layout(mi),*f,VARIANTS[v],seed=sd,weights=make_weights(400),valid_walks=_walks(mi,400,500)); log=s.run(7200); n=len(log)
    return a,(sum(r["mean_idleness"] for r in log)/n,sum(r["ghost_mean_idleness"] for r in log)/n)
if __name__=="__main__":
    jobs=[(mi,f,v,s) for mi in (0,1) for f in FLEETS for v in VS for s in range(10)]
    with Pool() as p: res=p.map(job,jobs)
    d=collections.defaultdict(dict)
    with open("reduced_results.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["map","fleet","variant","seed","road","air_mean"])
        for (mi,f,v,s),(g,h) in res:
            d[(mi,f,v)][s]=g+h; w.writerow([f"M{mi+1}",f"{f[0]}-{f[1]}",v,s,round(g,2),round(h,2)])
    for mi in (0,1):
        for f in FLEETS:
            full=d[(mi,f,"full")]; line=f"M{mi+1} {f[0]}-{f[1]}: full {st.mean(full.values()):7.1f}"
            for v in VS[1:4]:
                x=d[(mi,f,v)]; diff=[x[s]-full[s] for s in full]; m=st.mean(diff); sd=st.stdev(diff)
                line+=f" | {v[:7]} {st.mean(x.values()):7.1f} ({m:+6.1f}, t={m/(sd/math.sqrt(10)) if sd else 0:5.2f}, full lower {sum(q>0 for q in diff)}/10)"
            print(line)
