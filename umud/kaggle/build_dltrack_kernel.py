"""Build the DL_Track_US 0.3.1 re-run Kaggle kernel (prize-eligible member, EXP-2401).

    python umud/kaggle/build_dltrack_kernel.py --out umud/build/dltrack
    kaggle kernels push -p umud/build/dltrack

The kernel (CPU, internet on, competition data attached):
1. downloads the official DL_Track_US v0.3.1 release zip from OSF (node 7mjsc, Apache-2.0) and extracts the .h5 models;
2. clones DL_Track_US tag v0.3.1 from GitHub;
3. runs the DLTrack apo+fascicle VGG16 U-Nets and DLTrack's own geometry (doCalculations, default settings,
   bar scaling) on all 309 test images; images whose scale bars are not detected fall back to the umud
   tick-mark scale parser (re-runnable, part of this repo);
4. writes dltrack_rerun.csv (PA/FL/MT in mm, per-image medians + means), probability masks and a run record.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KERNEL = r'''
# umud EXP-2401: DL_Track_US 0.3.1 re-run (prize-eligible member)
# Apache-2.0 models by the competition host (Paul Ritsche); DLTrack geometry defaults; umud scale fallback.
import os, sys, glob, json, subprocess, time, shutil
import numpy as np
import pandas as pd
import cv2

OUT = "/kaggle/working"
os.makedirs(f"{OUT}/masks", exist_ok=True)
t0 = time.time()

# ---- 1. DLTrack code (tag v0.3.1) ----
REPO = "/kaggle/tmp/dltrack_repo"
if not os.path.isdir(REPO):
    os.makedirs("/kaggle/tmp", exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "-b", "v0.3.1",
                    "https://github.com/PaulRitsche/DL_Track_US.git", REPO], check=True)
sys.path.insert(0, REPO)  # repo root CONTAINS the DL_Track_US package

# ---- 2. weights from the official OSF v0.3.1 release ----
H5_DIR = "/kaggle/tmp/models"
if not glob.glob(H5_DIR + "/*.h5"):
    os.makedirs(H5_DIR, exist_ok=True)
    import requests
    r = requests.get("https://api.osf.io/v2/nodes/7mjsc/files/osfstorage/696922dd88da3924d54f21be/", timeout=60)
    dl = None
    for it in r.json()["data"]:
        if it["attributes"]["name"].lower().endswith(".zip"):
            cand = it["attributes"]
            print("zip candidate:", cand["name"], round(cand.get("size", 0) / 1e6, 1), "MB", flush=True)
            dl = it["links"]["download"] if cand.get("size", 0) > 1e8 else dl
    assert dl, "OSF v0.3.1 zip not found"
    zp = "/kaggle/tmp/dltrack.zip"
    with requests.get(dl, stream=True, timeout=1800) as resp:
        resp.raise_for_status()
        with open(zp, "wb") as fh:
            n = 0
            for chunk in resp.iter_content(1 << 22):
                fh.write(chunk); n += len(chunk)
                if n % (1 << 26) < (1 << 22): print("  downloaded", round(n / 1e6), "MB", flush=True)
    print("zip", round(os.path.getsize(zp) / 1e6, 1), "MB", flush=True)
    subprocess.run(["unzip", "-q", "-o", zp, "-d", "/kaggle/tmp/x"], check=True)
    h5s = glob.glob("/kaggle/tmp/x/**/*.h5", recursive=True)
    assert h5s, f"no .h5 weights in the release zip: {glob.glob('/kaggle/tmp/x/*')[:5]}"
    for p in h5s:
        shutil.copy(p, H5_DIR)
print("models:", sorted(os.path.basename(p) for p in glob.glob(H5_DIR + "/*.h5")), flush=True)

def pick(names, key):
    for n in names:
        if key in n.lower(): return n
    return None

h5s = sorted(os.path.basename(p) for p in glob.glob(H5_DIR + "/*.h5"))
apo_name = pick(h5s, "apo") or h5s[0]
img_fasc = [n for n in h5s if "fasc" in n.lower() and "ifss" not in n.lower()]  # IFSS = video model; images use VGG16
fasc_name = pick(img_fasc, "fasc-vgg") or (img_fasc[0] if img_fasc else pick(h5s, "vgg") or h5s[-1])
print("apo model:", apo_name, "| fasc model:", fasc_name, flush=True)

# ---- 3. DLTrack imports ----
from DL_Track_US.gui_helpers import calculate_architecture as ca
from DL_Track_US.gui_helpers.do_calculations import doCalculations
from DL_Track_US.gui_helpers.calibrate import calibrateDistanceStatic

# umud scale fallback (the repo's tick-mark parser; also committed with the final repo)
import urllib.request
sc_url = "https://raw.githubusercontent.com/Hisernberg/kraggle-3-in-one/claude/umud-muscle-architecture-qitgev/umud/umud/scale.py"
open("/kaggle/tmp/umud_scale.py", "w").write(urllib.request.urlopen(sc_url, timeout=60).read().decode())
sys.path.insert(0, "/kaggle/tmp")
import umud_scale as us

# ---- 4. data ----
DATA = sorted(glob.glob("/kaggle/input/**/test_images_v2/test_set_v2", recursive=True))[0]
files = sorted(glob.glob(DATA + "/*.tif"))
print("test images:", len(files), flush=True)

SETTINGS = json.load(open(REPO + "/DL_Track_US/gui_helpers/gui_files/settings.json"))
model_apo = ca.tf.keras.models.load_model(
    f"{H5_DIR}/{apo_name}", custom_objects={"IoU": ca.IoU, "dice_bce_loss": ca.dice_bce_loss, "dice_score": ca.dice_score})
model_fasc = ca.tf.keras.models.load_model(
    f"{H5_DIR}/{fasc_name}", custom_objects={"IoU": ca.IoU, "dice_bce_loss": ca.dice_bce_loss, "dice_score": ca.dice_score})

SPACING = 10
rows, failures = [], []
for i, path in enumerate(files):
    fid = os.path.basename(path)
    try:
        img, img_copy, nonflipped, h, w, filename = ca.importAndReshapeImage(path, 0)
        # probability masks for the record (img IS the model input: (1,512,512,3), normalized)
        pa_po = model_apo.predict(img, verbose=0)
        pf_po = model_fasc.predict(img, verbose=0)
        a8 = (pa_po[0, ..., 0] * 255).astype("uint8") if pa_po.ndim == 4 else (np.squeeze(pa_po) * 255).astype("uint8")
        f8 = (pf_po[0, ..., 0] * 255).astype("uint8") if pf_po.ndim == 4 else (np.squeeze(pf_po) * 255).astype("uint8")
        cv2.imwrite(f"{OUT}/masks/{fid}_apo.png", cv2.resize(a8, (w, h), interpolation=cv2.INTER_LINEAR))
        cv2.imwrite(f"{OUT}/masks/{fid}_fasc.png", cv2.resize(f8, (w, h), interpolation=cv2.INTER_LINEAR))
        # DLTrack bar scaling; umud tick-mark fallback when the bars are not detected
        calib_px, statement = calibrateDistanceStatic(nonflipped, SPACING)
        src = "bar"
        if calib_px is None:
            sc = us.detect_scale(nonflipped, os.path.splitext(fid)[1])
            calib_px, src = sc.px_per_mm * SPACING, f"umud_{sc.family}"
        fasc_l, pennation, x_low, x_high, midthick, fig = doCalculations(
            original_image=img, img_copy=img_copy, h=h, w=w, calib_dist=calib_px,
            spacing=SPACING, model_apo=model_apo, model_fasc=model_fasc,
            dictionary=SETTINGS, filter_fasc=False, image_callback=None)
        fl = np.asarray(fasc_l, dtype=float).ravel()
        pa = np.asarray(pennation, dtype=float).ravel()
        mt = float(np.asarray(midthick).ravel()[0]) if np.asarray(midthick).size else np.nan
        pxmm = float(calib_px) / SPACING
        if src == "bar":
            fl_mm, mt_mm = fl * 10.0, mt * 10.0
        else:
            fl_mm = fl / pxmm
            mt_mm = mt / pxmm if np.isfinite(mt) else np.nan
        ok_fl = fl.size and np.isfinite(fl_mm).any()
        rows.append(dict(image_id=fid,
                         pa_deg=float(np.nanmedian(pa)) if pa.size else np.nan,
                         fl_mm=float(np.nanmedian(fl_mm[ok_fl])) if ok_fl else np.nan,
                         fl_mean_mm=float(np.nanmean(fl_mm[ok_fl])) if ok_fl else np.nan,
                         mt_mm=mt_mm, n_fasc=int(pa.size),
                         calib_px_per_10mm=float(calib_px), src=src))
    except Exception as e:
        failures.append((fid, repr(e)[:200]))
    if i % 20 == 0:
        print(i, round(time.time() - t0, 1), "s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/dltrack_rerun.csv", index=False)
json.dump({"settings": SETTINGS, "spacing_mm": SPACING, "apo_model": apo_name, "fasc_model": fasc_name,
           "n_ok": len(rows), "failures": failures}, open(f"{OUT}/run_record.json", "w"), indent=1)
print("done:", len(rows), "ok,", len(failures), "failed", flush=True)
print(df.describe().T.to_string(), flush=True)
'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="kragglenote2forwork")
    ap.add_argument("--out", default=str(ROOT.parent / "build" / "dltrack"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel.py").write_text(KERNEL)
    meta = {
        "id": f"{a.user}/umud-dltrack-rerun",
        "title": "umud-dltrack-rerun",
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "competition_sources": ["umud-challenge-muscle-architecture-in-ultrasound-data"],
        "dataset_sources": [],
        "kernel_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)

if __name__ == "__main__":
    main()
