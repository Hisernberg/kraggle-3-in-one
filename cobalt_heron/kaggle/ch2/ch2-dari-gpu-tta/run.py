"""ch2 GPU map generator (D4 TTA) for the public dariushafshar U-Net set (no labels read).
JOBS env-like constant below: list of (split, family, tta). split in {test, oof, verify}.
oof: fold-k model on fold-k train images (folds.csv). test: mean of the 5 fold models (uint8 per model, as the author).
verify: resnet34 fold-0 model on 40 seeded non-fold-0 (in-train) images -> fold-honesty check.
Output: <split>_<family>.tar of uint8 PNG maps at 2048."""
import glob, hashlib, importlib, json, os, subprocess, sys, time, tarfile
import numpy as np, pandas as pd
JOBS = [("test","r34",4), ("test","eff",4), ("oof","r34",4), ("oof","eff",4)]
T0 = time.time(); INP, OUT = "/kaggle/input", "/kaggle/working"
def ff(name):
    h = sorted(glob.glob(os.path.join(INP, "**", name), recursive=True)); assert len(h) == 1, (name, h[:3]); return h[0]
RP = ff("solar_unet_receipt.json"); RECEIPT = json.load(open(RP)); WD = os.path.dirname(RP)
try:
    import segmentation_models_pytorch as smp
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", ff("segmentation_models_pytorch-0.5.0-py3-none-any.whl")], check=True)
    importlib.invalidate_caches(); import segmentation_models_pytorch as smp
import torch, cv2
from PIL import Image
torch.set_num_threads(os.cpu_count()); DEV = "cuda" if torch.cuda.is_available() else "cpu"; print("cpus", os.cpu_count(), torch.__version__, DEV, flush=True)
H = W = 2048; DISK_R = 950
YY, XX = np.ogrid[:H, :W]; DISK = ((YY - (H - 1) / 2.0) ** 2 + (XX - (W - 1) / 2.0) ** 2) <= DISK_R ** 2
TR = [d for d in glob.glob(INP + "/**/train_images", recursive=True) if os.path.isdir(d)][0]
TE = [d for d in glob.glob(INP + "/**/test_images", recursive=True) if os.path.isdir(d)][0]
folds = pd.read_csv(ff("folds.csv"))
test_stems = sorted(os.path.basename(p)[:-5] for p in glob.glob(TE + "/*.jpeg"))
def load(path):
    with Image.open(path) as im: return np.asarray(im.convert("L"), dtype=np.uint8)
D4 = [(0, False), (0, True), (2, False), (2, True), (1, False), (1, True), (3, False), (3, True)]
def _tf(z, k, fl):
    if fl: z = torch.flip(z, dims=[3])
    return torch.rot90(z, k, dims=[2, 3]) if k else z
def _itf(p, k, fl):
    if k: p = torch.rot90(p, -k, dims=[2, 3])
    return torch.flip(p, dims=[3]) if fl else p
@torch.inference_mode()
def predict(model, img, tta, cl):
    v = img[DISK].astype(np.float32) / 255.0; m, s = v.mean(), max(float(v.std()), 1e-3)
    z = ((torch.from_numpy(img.copy()).float().to(DEV) / 255.0 - float(m)) / float(s))[None, None]
    acc = torch.zeros((H, W), dtype=torch.float32, device=DEV)
    for k, fl in D4[:tta]:
        inp = _tf(z, k, fl)
        if cl: inp = inp.contiguous(memory_format=torch.channels_last)
        acc += _itf(torch.sigmoid(model(inp).float()), k, fl)[0, 0]
    acc /= tta
    return (acc * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().numpy()
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def fam(m): return "r34" if "resnet34" in m["encoder"] else "eff"
def load_model(m):
    assert sha(os.path.join(WD, m["file"])) == m["sha256"], m["file"]
    net = smp.Unet(encoder_name=m["encoder"], encoder_weights=None, in_channels=1, classes=1)
    net.load_state_dict(torch.load(os.path.join(WD, m["file"]), map_location="cpu", weights_only=True))
    cl = fam(m) == "r34"
    if cl: net = net.to(memory_format=torch.channels_last)
    return net.to(DEV).eval(), cl
models = RECEIPT["ensemble"]["models"]
log = []
for split, family, tta in JOBS:
    d = os.path.join(OUT, split, family); os.makedirs(d, exist_ok=True)
    ms = sorted([m for m in models if fam(m) == family], key=lambda m: int(m["fold"]))
    if split == "test":
        acc = np.zeros((len(test_stems), H, W), np.uint16)
        for m in ms:
            net, cl = load_model(m); t = time.time()
            for i, s in enumerate(test_stems):
                acc[i] += predict(net, load(os.path.join(TE, s + ".jpeg")), tta, cl)
            msg = f"test {m['file']} {time.time()-t:.0f}s total {(time.time()-T0)/60:.1f}min"; print(msg, flush=True); log.append(msg)
        for i, s in enumerate(test_stems):
            cv2.imwrite(os.path.join(d, s + ".png"), np.clip(np.round(acc[i] / len(ms)), 0, 255).astype(np.uint8))
        del acc
    elif split == "oof":
        for m in ms:
            net, cl = load_model(m); t = time.time(); st = folds[folds.fold == int(m["fold"])].stem.tolist()
            for s in st:
                cv2.imwrite(os.path.join(d, s + ".png"), predict(net, load(os.path.join(TR, s + ".jpeg")), tta, cl))
            msg = f"oof {m['file']} {len(st)} imgs {time.time()-t:.0f}s total {(time.time()-T0)/60:.1f}min"; print(msg, flush=True); log.append(msg)
    elif split == "verify":
        m = [m for m in ms if int(m["fold"]) == 0][0]; net, cl = load_model(m)
        rng = np.random.RandomState(0); pick = sorted(rng.choice(folds[folds.fold != 0].stem.tolist(), 40, replace=False).tolist())
        for s in pick:
            cv2.imwrite(os.path.join(d, s + ".png"), predict(net, load(os.path.join(TR, s + ".jpeg")), tta, cl))
        log.append(f"verify done {(time.time()-T0)/60:.1f}min")
    with tarfile.open(os.path.join(OUT, f"{split}_{family}.tar"), "w") as tf: tf.add(d, arcname=f"{split}/{family}")
    subprocess.run(["rm", "-rf", d])
json.dump({"jobs": JOBS, "log": log, "minutes": (time.time() - T0) / 60}, open(os.path.join(OUT, "run.json"), "w"))
print("done", (time.time() - T0) / 60, flush=True)
