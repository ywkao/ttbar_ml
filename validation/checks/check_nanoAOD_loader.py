import sys
from pathlib import Path

# 這個檔 -> checks/ -> validation/ -> ttbar_ml/  (往上 3 個 parent)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
print(f"check: {PROJECT_ROOT}")

from validation.nanoAOD_loader import load
file = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
s = load([file], nevents=2000, with_weights=True)
print(len(s["features"]), s["features"]["lep_pt"].shape, s["features"]["lep_pt"].dtype)
print(len(s["weights"]), s["weights"]["sm_point"].shape, s["weights"]["sm_point"].dtype)
