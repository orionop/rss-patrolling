import sys; sys.argv=[sys.argv[0]]
from big import *
import multiprocessing as mp
if __name__=="__main__":
    walks(0); walks(1)   # build once in the parent; forked workers inherit the cache
    VS=("full","no_ghost_credit","fixed_launch_nodes","nearest_ugv_recovery","tier0_ugv_only")
    jobs=[(i,v,s,7200) for i in (0,1) for v in VS for s in range(10)]
    ctx=mp.get_context("fork"); t=time.time(); out=[]
    with ctx.Pool() as p:
        for k,r in enumerate(p.imap_unordered(job,jobs),1):
            out.append(r)
            if k%10==0: print(k,"/",len(jobs),round(time.time()-t),"s",flush=True)
    with open("big_results.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["map","variant","seed","road","air_mean","air_max"])
        for (i,v,s,_),(a,b,c) in out: w.writerow(["L1" if i==0 else "L2",v,s,a,b,c])
    print("done",round(time.time()-t))
