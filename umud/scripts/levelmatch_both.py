"""Build a card with run.py, then iterate per-family PA finals and FL scales so every device's mean final PA and FL
equal the current best CSV's (state.json best_name). Usage: levelmatch_both.py NAME [extra blend args]"""
import subprocess, sys, json, pandas as pd
U = '/home/urad/work/k3/eb/umud'; name = sys.argv[1]; extra = sys.argv[2:]
st = json.load(open(f'{U}/daily/state.json'))
base = pd.read_csv(f"{U}/submissions/{st['best_name']}.csv").set_index('image_id')
fe = pd.read_csv(f'{U}/daily/inputs/features_ens.csv').set_index('image_id').loc[base.index]
FAMS = ['juniper_right', 'gray853', 'h512', 'telemed_644', 'left_ticks_800', 'lumify_png']
fam = fe['family'].astype(str).map(lambda s: next(k for k in FAMS if s.startswith(k)))
a = st['best_args']
pa = {a[i+1]: float(a[i+2]) for i, t in enumerate(a) if t == '--fam-pa-final'}
fl = {a[i+1]: float(a[i+2]) for i, t in enumerate(a) if t == '--fam-fl-scale'}
def run():
    ex = list(extra)
    for k, v in pa.items(): ex += ['--fam-pa-final', k, f'{v:.4f}']
    for k, v in fl.items(): ex += ['--fam-fl-scale', k, f'{v:.5f}']
    r = subprocess.run([sys.executable, 'daily/run.py', 'build', name, *ex], cwd=U, capture_output=True, text=True)
    if r.returncode: print(r.stderr[-2000:]); sys.exit(1)
    return pd.read_csv(f'{U}/daily/out/{name}.csv').set_index('image_id').loc[base.index]
for it in range(5):
    o = run()
    g = pd.DataFrame({'f': fam, 'po': o.pa_deg, 'pb': base.pa_deg, 'fo': o.fl_mm, 'fb': base.fl_mm}).groupby('f').mean()
    print(it, 'dPA', (g.po-g.pb).round(4).to_dict(), 'dFLrel', (g.fo/g.fb-1).round(5).to_dict())
    for k in pa: pa[k] += g.loc[k, 'pb'] - g.loc[k, 'po']
    for k in fl: fl[k] *= g.loc[k, 'fb'] / g.loc[k, 'fo']
o = run()
for t in ['pa_deg', 'fl_mm', 'mt_mm']:
    d = o[t] - base[t]
    print(t, 'rows moved', int((d.abs() > 1e-6).sum()), 'mean|d|', round(d.abs().mean(), 3), 'mean', round(d.mean(), 4))
    print('   per fam mean|d|', d.abs().groupby(fam).mean().round(3).to_dict())
