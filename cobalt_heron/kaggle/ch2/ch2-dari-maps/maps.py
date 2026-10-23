"""ch2-dari-maps: 2048 probability maps from the public dariushafshar 10-model U-Net set.
OOF: fold-k models (R34 + EffV2-S) predict fold-k train images (folds.csv) -> oof/{fam}/{stem}.png
Test: all 10 models -> test/{fam}/{stem}.png (family mean, uint8)
Verify: r34 fold0 on 40 non-fold-0 (in-train) images -> verify/{stem}.png (fold-honesty check)
No labels are read here. D4 TTA, uint8 rounding per model, as in the author's notebook."""
import glob, hashlib, importlib, json, os, subprocess, sys, threading, time
import numpy as np, pandas as pd
T0 = time.time(); INP, OUT = "/kaggle/input", "/kaggle/working"
def ff(name):
    h = sorted(glob.glob(os.path.join(INP, "**", name), recursive=True)); assert len(h) == 1, (name, h[:3]); return h[0]
RECEIPT = json.load(open(ff("solar_unet_receipt.json"))); WD = os.path.dirname(ff("solar_unet_receipt.json"))
try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", ff("segmentation_models_pytorch-0.5.0-py3-none-any.whl")], check=True)
    importlib.invalidate_caches(); import segmentation_models_pytorch as smp
import torch, cv2
from PIL import Image
H = W = 2048; DISK_R = 950
YY, XX = np.ogrid[:H, :W]; DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2
TR = [d for d in glob.glob(INP + "/**/train_images", recursive=True) if os.path.isdir(d)][0]
TE = [d for d in glob.glob(INP + "/**/test_images", recursive=True) if os.path.isdir(d)][0]
folds = pd.read_csv(ff("folds.csv")); print(folds.fold.value_counts().to_dict(), TR, TE, flush=True)
test_stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(TE + "/*.jpeg"))
def load(path):
    with Image.open(path) as im: return np.asarray(im.convert("L"), dtype=np.uint8)
def stats(img):
    v = img[DISK].astype(np.float32) / 255.0; return v.mean(), max(float(v.std()), 1e-3)
D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]
def _tf(z, k, fl):
    if fl: z = torch.flip(z, dims=[3])
    return torch.rot90(z, k, dims=[2, 3]) if k else z
def _itf(p, k, fl):
    if k: p = torch.rot90(p, -k, dims=[2, 3])
    return torch.flip(p, dims=[3]) if fl else p
@torch.no_grad()
def predict(model, img, dev, vpp=4):
    m, s = stats(img)
    x = torch.from_numpy(img.copy()).to(dev).float().div_(255.0); z = ((x - float(m)) / float(s))[None, None]
    acc = torch.zeros((H, W), dtype=torch.float32, device=dev)
    for j in range(0, 8, vpp):
        ch = D4[j:j + vpp]; inp = torch.cat([_tf(z, k, fl) for k, fl in ch]).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type="cuda", dtype=torch.float16): out = model(inp)
        pr = torch.sigmoid(out.float())
        for q, (k, fl) in enumerate(ch): acc += _itf(pr[q:q + 1], k, fl)[0, 0]
    acc /= 8
    return (acc * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
models = RECEIPT["ensemble"]["models"]
print(json.dumps(models[:2])[:400], flush=True)
def fam(m): return "r34" if "resnet34" in m["encoder"] else "eff"
jobs = []
rng = np.random.RandomState(0)
for m in models:
    k = int(m["fold"]); st = folds[folds.fold == k].stem.tolist()
    jobs.append((m, [(s, os.path.join(TR, s + ".jpeg"), f"oof/{fam(m)}/{s}.png") for s in st]))
    jobs.append((m, [(s, os.path.join(TE, s + ".jpeg"), ("test", fam(m), i)) for i, s in enumerate(test_stems)]))
    if fam(m) == "r34" and k == 0:
        other = folds[folds.fold != 0].stem.tolist(); pick = sorted(rng.choice(other, 40, replace=False).tolist())
        jobs.append((m, [(s, os.path.join(TR, s + ".jpeg"), f"verify/r34f0/{s}.png") for s in pick]))
jobs.sort(key=lambda j: 0 if fam(j[0]) == "eff" else 1)
for d in ["oof/r34", "oof/eff", "test/r34", "test/eff", "verify/r34f0"]: os.makedirs(os.path.join(OUT, d), exist_ok=True)
acc = {f: np.zeros((len(test_stems), H, W), np.uint16) for f in ("r34", "eff")}; cnt = {"r34": 0, "eff": 0}
lock = threading.Lock(); errors = []; logs = []
def worker(dev):
    cur = None; model = None
    while True:
        with lock:
            if not jobs or errors: return
            m, items = jobs.pop(0)
        try:
            t = time.time()
            if cur != m["file"]:
                assert sha(os.path.join(WD, m["file"])) == m["sha256"], m["file"]
                model = smp.Unet(encoder_name=m["encoder"], encoder_weights=None, in_channels=1, classes=1)
                model.load_state_dict(torch.load(os.path.join(WD, m["file"]), map_location="cpu", weights_only=True))
                model = model.to(dev).to(memory_format=torch.channels_last).eval(); cur = m["file"]
            for s, path, key in items:
                p = predict(model, load(path), dev)
                if isinstance(key, tuple):
                    with lock: acc[key[1]][key[2]] += p
                else:
                    cv2.imwrite(os.path.join(OUT, key), p, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            if isinstance(items[0][2], tuple):
                with lock: cnt[fam(m)] += 1
            msg = f"{m['file']} {dev} {len(items)} imgs {time.time()-t:.0f}s total {(time.time()-T0)/60:.1f}min left {len(jobs)}"
            print(msg, flush=True); logs.append(msg)
        except Exception as e:
            import traceback; traceback.print_exc(); errors.append(repr(e)); return
torch.backends.cudnn.benchmark = True
devs = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
print("GPUs", [torch.cuda.get_device_name(i) for i in range(len(devs))], flush=True)
th = [threading.Thread(target=worker, args=(d,)) for d in devs]
[t.start() for t in th]; [t.join() for t in th]
assert not errors, errors
for f in ("r34", "eff"):
    assert cnt[f] == 5, cnt
    for i, s in enumerate(test_stems):
        cv2.imwrite(os.path.join(OUT, f"test/{f}/{s}.png"), np.clip(np.round(acc[f][i] / cnt[f]), 0, 255).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 6])
json.dump({"logs": logs, "minutes": (time.time() - T0) / 60}, open(os.path.join(OUT, "run.json"), "w"))
# pack to reduce file count
import tarfile
for d in ["oof", "test", "verify"]:
    with tarfile.open(os.path.join(OUT, f"{d}.tar"), "w") as tf: tf.add(os.path.join(OUT, d), arcname=d)
    subprocess.run(["rm", "-rf", os.path.join(OUT, d)])
print("done", (time.time() - T0) / 60, "min", flush=True)
