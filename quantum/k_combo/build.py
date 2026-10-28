# Account B (kragglenote2forwork) - Quantum combo v1
# Combines two independent, public, credited improvements on SomeJoint v16 full-XOR C60 chemistry (public 22.62971):
#  - Atakan Aldemir, "C60 Reachable Cache" (angle-loader circuits, T 16528 -> 16464), kernel output submission.zip
#  - Kirill Tymchishin, "Exact Gauge + 5-Word QROM" (orthogonal column rotation of W/H2_I, lambda_eff 48.561 -> 48.517)
# Our change: Atakan's 15 loader circuits + invsqrt + dqi, with Kirill's rotated factor NPZ (angle_words etc. byte-identical).
# Then the unchanged official chemistry verifier (full default shots) checks the combined bytes.
import glob, hashlib, io, json, math, os, sys, zipfile, shutil, pathlib
import numpy as np
INP = '/kaggle/input'
def find(pattern, must):
    hits = [h for h in glob.glob(f'{INP}/**/{pattern}', recursive=True) if must in h]
    assert hits, (pattern, must, glob.glob(f'{INP}/**/{pattern}', recursive=True)); return hits[0]
A = find('submission.zip', 'reachable-cache')
K = find('submission.zip', 'exact-gauge')
COMP = os.path.dirname(os.path.dirname(find('verify_submission_invsqrt.py', 'verifiers')))
COMP = os.path.dirname(COMP) if not os.path.isdir(os.path.join(COMP, 'kickmix')) else COMP
print('A', A, '\nK', K, '\nCOMP', COMP)
sys.path[:0] = [os.path.join(COMP, 'kickmix'), os.path.join(COMP, 'verifiers')]
import kickmix as km
from dfthcsos import verify as vdft
sha = lambda b: hashlib.sha256(b).hexdigest()
za, zk = zipfile.ZipFile(A), zipfile.ZipFile(K)
assert sorted(za.namelist()) == sorted(zk.namelist())
na = np.load(io.BytesIO(za.read('dfthcsos/submission.npz'))); nk = np.load(io.BytesIO(zk.read('dfthcsos/submission.npz')))
for n in na.files:
    if n not in ('W', 'H2_I'): assert np.array_equal(na[n], nk[n]), n
SUB = pathlib.Path('/kaggle/tmp/sub'); shutil.rmtree(SUB, ignore_errors=True); (SUB/'dfthcsos').mkdir(parents=True)
members = {}
for n in za.namelist():
    b = zk.read(n) if n == 'dfthcsos/submission.npz' else za.read(n)
    (SUB/n).write_bytes(b); members[n] = sha(b)
res = vdft.verify(str(SUB/'dfthcsos'), salt='acctb-combo-v1')
print({k: res.get(k) for k in ('status','score','lambda_eff','toffoli_count','num_qubits')})
assert res['status'] == 'PASS', res
m = res['metrics']; print('error_1', m.get('error_1'), 'error_2', m.get('error_2'), 'gap', m.get('sos_gap', m.get('gap')))
# Atakan same-salt-independent receipts for unchanged invsqrt/dqi bytes: GM 22.659008 with chem 343787901.226
gm = 22.659007681666946 * (343787901.22637963 / float(res['score'])) ** (1/3)
print('combined local GM estimate', gm)
out = '/kaggle/working/submission.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
    for n in sorted(members): z.write(SUB/n, n)
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None and {n: sha(z.read(n)) for n in z.namelist()} == members
json.dump({'chem': {k: res.get(k) for k in ('status','score','lambda_eff','toffoli_count','num_qubits')}, 'gm_est': gm, 'members': members},
          open('/kaggle/working/combo_audit.json', 'w'), indent=1, default=float)
print('DONE', gm)
