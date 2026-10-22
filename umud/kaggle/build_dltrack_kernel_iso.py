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
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # the .h5 weights are TF2.x/Keras2; Keras 3 rejects them
import sys, glob, json, subprocess, time, shutil
import numpy as np
import pandas as pd
import cv2

OUT = "/kaggle/working"
os.makedirs(f"{OUT}/masks", exist_ok=True)
t0 = time.time()

# ---- 0. deps: legacy keras (h5 loading) + GUI deps needed at import time ----
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tf-keras", "customtkinter", "darkdetect"], check=True)

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

# ---- 3. DLTrack imports (bypass the package __init__: it drags in GUI deps) ----
import types
import importlib.util

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

pkg = types.ModuleType("DL_Track_US"); pkg.__path__ = [REPO + "/DL_Track_US"]; sys.modules["DL_Track_US"] = pkg
gh = types.ModuleType("DL_Track_US.gui_helpers"); gh.__path__ = [REPO + "/DL_Track_US/gui_helpers"]; sys.modules["DL_Track_US.gui_helpers"] = gh
ca = _load("DL_Track_US.gui_helpers.calculate_architecture", REPO + "/DL_Track_US/gui_helpers/calculate_architecture.py")
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
files = sorted(glob.glob(DATA + "/*.tif") + glob.glob(DATA + "/*.png"))
print("test images:", len(files), flush=True)

PXMM = {"IMG_00001.tif": 13.5333, "IMG_00002.tif": 15.225, "IMG_00003.tif": 13.5333, "IMG_00004.tif": 13.5333, "IMG_00005.tif": 10.15, "IMG_00006.tif": 10.15, "IMG_00007.tif": 13.5333, "IMG_00008.tif": 13.5333, "IMG_00009.tif": 15.9429, "IMG_00010.tif": 15.225, "IMG_00011.tif": 15.225, "IMG_00012.tif": 15.225, "IMG_00013.tif": 8.7, "IMG_00014.tif": 11.0727, "IMG_00015.tif": 10.15, "IMG_00016.tif": 15.225, "IMG_00017.tif": 15.225, "IMG_00018.tif": 9.3692, "IMG_00019.tif": 9.3692, "IMG_00020.tif": 15.225, "IMG_00021.tif": 15.225, "IMG_00022.tif": 12.18, "IMG_00023.tif": 12.18, "IMG_00024.tif": 15.225, "IMG_00025.tif": 15.225, "IMG_00026.tif": 15.225, "IMG_00027.tif": 15.225, "IMG_00028.tif": 15.225, "IMG_00029.tif": 15.225, "IMG_00030.tif": 15.225, "IMG_00031.tif": 15.225, "IMG_00032.tif": 15.225, "IMG_00033.tif": 15.225, "IMG_00034.tif": 12.18, "IMG_00035.tif": 15.225, "IMG_00036.tif": 16.68, "IMG_00037.tif": 16.7, "IMG_00038.tif": 16.68, "IMG_00039.tif": 16.68, "IMG_00040.tif": 7.78, "IMG_00041.tif": 7.78, "IMG_00042.tif": 7.78, "IMG_00043.tif": 7.78, "IMG_00044.tif": 7.78, "IMG_00045.tif": 7.78, "IMG_00046.tif": 7.78, "IMG_00047.tif": 16.68, "IMG_00048.tif": 7.78, "IMG_00049.tif": 16.68, "IMG_00050.tif": 16.68, "IMG_00051.tif": 16.68, "IMG_00052.tif": 16.68, "IMG_00053.tif": 16.68, "IMG_00054.tif": 16.68, "IMG_00055.tif": 16.68, "IMG_00056.tif": 12.61, "IMG_00057.tif": 12.61, "IMG_00058.tif": 12.61, "IMG_00059.tif": 12.61, "IMG_00060.tif": 12.61, "IMG_00061.tif": 12.61, "IMG_00062.tif": 12.61, "IMG_00063.tif": 12.61, "IMG_00064.tif": 12.61, "IMG_00065.tif": 12.61, "IMG_00066.tif": 14.82, "IMG_00067.tif": 14.82, "IMG_00068.tif": 14.82, "IMG_00069.tif": 14.82, "IMG_00070.tif": 14.82, "IMG_00071.tif": 14.82, "IMG_00072.tif": 14.82, "IMG_00073.tif": 14.82, "IMG_00074.tif": 14.82, "IMG_00075.tif": 14.82, "IMG_00076.tif": 14.82, "IMG_00077.tif": 14.82, "IMG_00078.tif": 14.82, "IMG_00079.tif": 14.82, "IMG_00080.tif": 14.82, "IMG_00081.tif": 14.82, "IMG_00082.tif": 14.82, "IMG_00083.tif": 14.82, "IMG_00084.tif": 14.82, "IMG_00085.tif": 14.82, "IMG_00086.tif": 14.82, "IMG_00087.tif": 14.82, "IMG_00088.tif": 14.82, "IMG_00089.tif": 14.82, "IMG_00090.tif": 14.82, "IMG_00091.tif": 14.82, "IMG_00092.tif": 14.82, "IMG_00093.tif": 14.82, "IMG_00094.tif": 14.82, "IMG_00095.tif": 14.82, "IMG_00096.tif": 14.82, "IMG_00097.tif": 14.82, "IMG_00098.tif": 14.82, "IMG_00099.tif": 14.82, "IMG_00100.tif": 14.82, "IMG_00101.tif": 14.82, "IMG_00102.tif": 14.82, "IMG_00103.tif": 14.82, "IMG_00104.tif": 14.82, "IMG_00105.tif": 14.82, "IMG_00106.tif": 12.61, "IMG_00107.tif": 12.61, "IMG_00108.tif": 12.61, "IMG_00109.tif": 12.61, "IMG_00110.tif": 12.61, "IMG_00111.tif": 12.61, "IMG_00112.tif": 12.61, "IMG_00113.tif": 12.61, "IMG_00114.tif": 12.61, "IMG_00115.tif": 12.61, "IMG_00116.tif": 12.61, "IMG_00117.tif": 12.61, "IMG_00118.tif": 12.61, "IMG_00119.tif": 12.61, "IMG_00120.tif": 12.61, "IMG_00121.tif": 12.61, "IMG_00122.tif": 12.61, "IMG_00123.tif": 12.61, "IMG_00124.tif": 12.61, "IMG_00125.tif": 12.61, "IMG_00126.tif": 12.61, "IMG_00127.tif": 12.61, "IMG_00128.tif": 12.61, "IMG_00129.tif": 12.61, "IMG_00130.tif": 12.61, "IMG_00131.tif": 12.61, "IMG_00132.tif": 12.61, "IMG_00133.tif": 12.61, "IMG_00134.tif": 12.61, "IMG_00135.tif": 12.61, "IMG_00136.tif": 12.61, "IMG_00137.tif": 12.61, "IMG_00138.tif": 12.61, "IMG_00139.tif": 12.61, "IMG_00140.tif": 12.61, "IMG_00141.tif": 12.61, "IMG_00142.tif": 12.61, "IMG_00143.tif": 12.61, "IMG_00144.tif": 12.61, "IMG_00145.tif": 12.61, "IMG_00146.tif": 14.82, "IMG_00147.tif": 14.82, "IMG_00148.tif": 14.82, "IMG_00149.tif": 14.82, "IMG_00150.tif": 14.82, "IMG_00151.tif": 14.82, "IMG_00152.tif": 14.82, "IMG_00153.tif": 14.82, "IMG_00154.tif": 14.82, "IMG_00155.tif": 14.82, "IMG_00156.tif": 14.82, "IMG_00157.tif": 14.82, "IMG_00158.tif": 14.82, "IMG_00159.tif": 14.82, "IMG_00160.tif": 14.82, "IMG_00161.tif": 14.82, "IMG_00162.tif": 14.82, "IMG_00163.tif": 14.82, "IMG_00164.tif": 14.82, "IMG_00165.tif": 14.82, "IMG_00166.tif": 14.82, "IMG_00167.tif": 14.82, "IMG_00168.tif": 14.82, "IMG_00169.tif": 14.82, "IMG_00170.tif": 14.82, "IMG_00171.tif": 14.82, "IMG_00172.tif": 14.82, "IMG_00173.tif": 14.82, "IMG_00174.tif": 14.82, "IMG_00175.tif": 14.82, "IMG_00176.tif": 14.82, "IMG_00177.tif": 14.82, "IMG_00178.tif": 14.82, "IMG_00179.tif": 14.82, "IMG_00180.tif": 14.82, "IMG_00181.tif": 14.82, "IMG_00182.tif": 14.82, "IMG_00183.tif": 14.82, "IMG_00184.tif": 14.82, "IMG_00185.tif": 14.82, "IMG_00186.tif": 14.82, "IMG_00187.tif": 14.82, "IMG_00188.tif": 14.82, "IMG_00189.tif": 14.82, "IMG_00190.tif": 14.82, "IMG_00191.tif": 14.82, "IMG_00192.tif": 14.82, "IMG_00193.tif": 14.82, "IMG_00194.tif": 14.82, "IMG_00195.tif": 14.82, "IMG_00196.tif": 15.225, "IMG_00197.tif": 12.18, "IMG_00198.tif": 15.9667, "IMG_00199.tif": 15.9667, "IMG_00200.tif": 15.9667, "IMG_00201.tif": 15.225, "IMG_00202.tif": 15.225, "IMG_00203.tif": 15.225, "IMG_00204.tif": 15.225, "IMG_00205.tif": 13.5333, "IMG_00206.tif": 13.5333, "IMG_00207.tif": 15.225, "IMG_00208.tif": 15.225, "IMG_00209.tif": 15.225, "IMG_00210.tif": 15.225, "IMG_00211.tif": 12.18, "IMG_00212.tif": 12.18, "IMG_00213.tif": 12.18, "IMG_00214.tif": 15.225, "IMG_00215.tif": 13.5333, "IMG_00216.tif": 13.5333, "IMG_00217.tif": 15.225, "IMG_00218.tif": 15.225, "IMG_00219.tif": 15.225, "IMG_00220.tif": 15.225, "IMG_00221.tif": 15.225, "IMG_00222.tif": 15.225, "IMG_00223.tif": 15.225, "IMG_00224.tif": 15.225, "IMG_00225.tif": 15.225, "IMG_00226.tif": 15.225, "IMG_00227.tif": 15.225, "IMG_00228.tif": 15.225, "IMG_00229.tif": 13.5333, "IMG_00230.tif": 13.5333, "IMG_00231.tif": 15.225, "IMG_00232.tif": 15.225, "IMG_00233.tif": 15.225, "IMG_00234.tif": 15.9429, "IMG_00235.tif": 15.9429, "IMG_00236.tif": 13.5333, "IMG_00237.tif": 12.18, "IMG_00238.tif": 15.225, "IMG_00239.tif": 15.225, "IMG_00240.tif": 15.225, "IMG_00241.tif": 15.225, "IMG_00242.tif": 12.18, "IMG_00243.tif": 15.225, "IMG_00244.tif": 15.225, "IMG_00245.tif": 15.225, "IMG_00246.tif": 15.225, "IMG_00247.tif": 15.225, "IMG_00248.tif": 15.225, "IMG_00249.tif": 15.225, "IMG_00250.tif": 15.225, "IMG_00251.tif": 15.9667, "IMG_00252.png": 15.0, "IMG_00253.png": 13.4, "IMG_00254.png": 17.2, "IMG_00255.png": 17.2, "IMG_00256.png": 17.2, "IMG_00257.png": 17.2, "IMG_00258.png": 12.1, "IMG_00259.png": 15.0, "IMG_00260.png": 17.2, "IMG_00261.png": 15.0, "IMG_00262.png": 17.2, "IMG_00263.png": 17.2, "IMG_00264.png": 15.0, "IMG_00265.png": 17.2, "IMG_00266.png": 15.0, "IMG_00267.png": 13.4, "IMG_00268.png": 13.4, "IMG_00269.png": 15.0, "IMG_00270.png": 17.2, "IMG_00271.png": 17.2, "IMG_00272.png": 17.2, "IMG_00273.png": 17.2, "IMG_00274.png": 13.4, "IMG_00275.png": 10.0, "IMG_00276.png": 17.2, "IMG_00277.png": 17.2, "IMG_00278.png": 15.0, "IMG_00279.png": 20.1, "IMG_00280.png": 20.1, "IMG_00281.png": 13.4, "IMG_00282.png": 17.2, "IMG_00283.png": 13.4, "IMG_00284.png": 13.4, "IMG_00285.png": 17.2, "IMG_00286.png": 17.2, "IMG_00287.png": 17.2, "IMG_00288.png": 12.1, "IMG_00289.png": 15.0, "IMG_00290.png": 15.0, "IMG_00291.png": 15.0, "IMG_00292.png": 13.4, "IMG_00293.png": 12.1, "IMG_00294.png": 12.1, "IMG_00295.png": 17.2, "IMG_00296.png": 12.1, "IMG_00297.png": 12.1, "IMG_00298.png": 12.1, "IMG_00299.png": 15.0, "IMG_00300.png": 15.0, "IMG_00301.png": 12.1, "IMG_00302.png": 15.0, "IMG_00303.png": 15.0, "IMG_00304.png": 12.1, "IMG_00305.png": 17.2, "IMG_00306.png": 17.2, "IMG_00307.png": 15.3, "IMG_00308.png": 15.0, "IMG_00309.png": 20.1}
SETTINGS = json.load(open(REPO + "/DL_Track_US/gui_helpers/gui_files/settings.json"))
model_apo = ca.tf.keras.models.load_model(
    f"{H5_DIR}/{apo_name}", custom_objects={"IoU": ca.IoU, "dice_bce_loss": ca.dice_bce_loss, "dice_score": ca.dice_score})
model_fasc = ca.tf.keras.models.load_model(
    f"{H5_DIR}/{fasc_name}", custom_objects={"IoU": ca.IoU, "dice_bce_loss": ca.dice_bce_loss, "dice_score": ca.dice_score})

SPACING = 10
import cv2
rows, failures = [], []
for i, path in enumerate(files):
    fid = os.path.basename(path)
    try:
        ppm = PXMM[fid]
        img0 = cv2.imread(path, 1)
        h0, w0 = img0.shape[:2]
        # B-mode crop via the umud scale parser (deterministic, re-runnable)
        sc = us.detect_scale(img0, os.path.splitext(fid)[1])
        crop = img0[sc.t:sc.b, sc.l:sc.r]
        ch, cw = crop.shape[:2]
        ppm_crop = sc.px_per_mm  # px per mm inside the crop (vertical ticks)
        # isotropic letterbox to 512x512
        s = 512.0 / max(ch, cw)
        nh, nw = max(1, int(round(ch * s))), max(1, int(round(cw * s)))
        img_r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        padded = np.zeros((512, 512, 3), dtype="float32")
        padded[:nh, :nw] = img_r
        x_in = np.reshape(padded, (1, 512, 512, 3))
        calib_px = None  # doCalculations' own list/float scaling crashes; convert units below (isotropic s known)
        pa_po = model_apo.predict(x_in, verbose=0)
        pf_po = model_fasc.predict(x_in, verbose=0)
        a8 = (pa_po[0, ..., 0] * 255).astype("uint8") if pa_po.ndim == 4 else (np.squeeze(pa_po) * 255).astype("uint8")
        f8 = (pf_po[0, ..., 0] * 255).astype("uint8") if pf_po.ndim == 4 else (np.squeeze(pf_po) * 255).astype("uint8")
        # de-letterbox masks back to original resolution for the record
        a8c = cv2.resize(a8[:nh, :nw], (cw, ch), interpolation=cv2.INTER_LINEAR)
        f8c = cv2.resize(f8[:nh, :nw], (cw, ch), interpolation=cv2.INTER_LINEAR)
        canvas_a = np.zeros((h0, w0), dtype="uint8"); canvas_f = np.zeros((h0, w0), dtype="uint8")
        canvas_a[sc.t:sc.b, sc.l:sc.r] = a8c; canvas_f[sc.t:sc.b, sc.l:sc.r] = f8c
        cv2.imwrite(f"{OUT}/masks/{fid}_apo.png", canvas_a)
        cv2.imwrite(f"{OUT}/masks/{fid}_fasc.png", canvas_f)
        fasc_l, pennation, x_low, x_high, midthick, fig = doCalculations(
            original_image=x_in, img_copy=padded, h=512, w=512, calib_dist=None,
            spacing=SPACING, model_apo=model_apo, model_fasc=model_fasc,
            dictionary=SETTINGS, filter_fasc=False, image_callback=None)
        fl_px = np.asarray(fasc_l, dtype=float).ravel()
        pa = np.asarray(pennation, dtype=float).ravel()
        mt_px = float(np.asarray(midthick).ravel()[0]) if np.asarray(midthick).size else np.nan
        pxmm_r = ppm_crop * s  # px per mm in the letterboxed space (isotropic by construction)
        fl_mm = fl_px / pxmm_r
        mt_mm = mt_px / pxmm_r if np.isfinite(mt_px) else np.nan
        ok_fl = fl_px.size and np.isfinite(fl_mm).any()
        rows.append(dict(image_id=fid,
                         pa_deg=float(np.nanmedian(pa)) if pa.size else np.nan,
                         fl_mm=float(np.nanmedian(fl_mm[ok_fl])) if ok_fl else np.nan,
                         fl_mean_mm=float(np.nanmean(fl_mm[ok_fl])) if ok_fl else np.nan,
                         mt_mm=mt_mm, n_fasc=int(pa.size), px_per_mm=ppm, family=sc.family, src="crop_let512"))
    except Exception as e:
        failures.append((fid, repr(e)[:200]))
    if i % 20 == 0:
        print(i, round(time.time() - t0, 1), "s", flush=True)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/dltrack_iso.csv", index=False)
json.dump({"settings": SETTINGS, "spacing_mm": SPACING, "apo_model": apo_name, "fasc_model": fasc_name,
           "n_ok": len(rows), "failures": failures}, open(f"{OUT}/run_record.json", "w"), indent=1)
print("done:", len(rows), "ok,", len(failures), "failed", flush=True)
print(df.describe().T.to_string(), flush=True) if len(df) else print("EMPTY", flush=True)
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
        "id": f"{a.user}/umud-dltrack-iso",
        "title": "umud-dltrack-iso",
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
