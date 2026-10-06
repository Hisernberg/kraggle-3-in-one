"""Build agents/annex/<name>/main.py = base agent source + annex wrapper.
    python agents/annex/build_annex.py public_agents/abhinav0370__cha22-agent/main.py cha22 [KEY=VAL ...]
"""
import os, sys
base_path, name = sys.argv[1], sys.argv[2]
overrides = dict(a.split("=", 1) for a in sys.argv[3:])
here = os.path.dirname(os.path.abspath(__file__))
src = open(base_path, encoding="utf-8").read()
core = open(os.path.join(here, "annex_core.py"), encoding="utf-8").read()
for k, v in overrides.items():
    core = core.replace(f'    "{k}": ', f'    "{k}": {v},  # override\n    "_old_{k}": ', 1)
out = src.rstrip("\n") + "\n\n\n# ======================= SE annex wrapper =======================\n" \
      "_base_agent = [_v for _v in list(globals().values()) if callable(_v)][-1]\n\n" + core
d = os.path.join(here, name)
os.makedirs(d, exist_ok=True)
open(os.path.join(d, "main.py"), "w", encoding="utf-8").write(out)
print(os.path.join(d, "main.py"))
