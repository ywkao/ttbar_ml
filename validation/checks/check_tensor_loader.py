import sys
from pathlib import Path

# this file -> checks/ -> validation/ -> ttbar_ml/  (3 parents up)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"check: {PROJECT_ROOT}")

from validation.tensor_loader import load
file = "/eos/uscms/store/user/ywkao/Outputs_sbi/pretraining/train.p"
s = load([file], with_weights=True)   # replace with the actual path to your train.p
print(len(s["features"]), s["features"]["lep_pt"].shape, s["features"]["lep_pt"].dtype)
print(len(s["weights"]), s["weights"]["sm_point"].shape, s["weights"]["sm_point"].dtype)
# expected: 41  (197871,)  float64
