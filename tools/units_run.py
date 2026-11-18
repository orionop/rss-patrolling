import sys,os,csv,multiprocessing as mp
HERE=os.path.dirname(os.path.abspath(__file__));SIM=os.path.join(HERE,"..","sim");sys.path.insert(0,SIM);os.chdir(SIM)
os.environ['GHOST_SEC']='1'
import runner
V=["full","no_ghost_credit","fixed_launch_nodes","nearest_ugv_recovery","tier0_ugv_only"]
jobs=[(m,n,f"M{m+1}",nu,na,v,s,7200,400,500) for m,n in [(0,0),(1,1)] for (nu,na) in [(4,2),(6,3)] for v in V for s in range(10)]
jobs=[(j[0],j[2],j[3],j[4],j[5],j[6],j[7],j[8],j[9]) for j in jobs]
if __name__=="__main__":
    rows=[]
    with mp.get_context("fork").Pool() as p:
        for r in p.imap_unordered(runner._job,jobs):
            rows.append(r);print(len(rows),len(jobs),flush=True)
    out=os.path.join(HERE,"..","results_extra","units_seconds_10seeds.csv")
    w=csv.DictWriter(open(out,"w",newline=""),fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);print("saved")
