import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sim'))
import runner, run_m2 as r
runner.run_map(1,"M2",[(8,4)],r.SEEDS,r.HORIZON_S,r.T_SECONDS,r.WALKS_PER_PAIR)   # writes results_m2.csv in the cwd
