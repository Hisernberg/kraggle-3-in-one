"""Official-metric validator. Usage: python validate.py submission.csv [n1,n2,...]
Runs the verbatim Kaggle metric (sis_metric.py) on all n (or the listed n only) and prints the score
plus any infeasible puzzles. Writes submission_details.csv (per-n L, 50 digits) in the cwd."""
import sys, os, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sis_metric as M
path = sys.argv[1]
ns = [int(a) for a in sys.argv[2].split(',')] if len(sys.argv) > 2 else list(range(1, 1001))
sub = pd.read_csv(path, dtype=str)
sub = sub[sub.n.astype(int).isin(ns)]
puzzles = M._parse_submission_csv(sub.drop(columns=['id']).to_csv(index=False))
total, results = M._evaluate_puzzles_parallel(puzzles, set(ns))
bad = [(n, r) for n, _, L, r in results if L is None]
print('SCORE', repr(float(total)), 'n_checked', len(ns), 'infeasible', len(bad))
for n, r in bad[:20]: print('  BAD', n, r)
