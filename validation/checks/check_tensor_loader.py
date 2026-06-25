import sys
from pathlib import Path

# 這個檔 -> checks/ -> validation/ -> ttbar_ml/  (往上 3 個 parent)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

print(f"check: {PROJECT_ROOT}")

from validation.tensor_loader import load
file = "/eos/uscms/store/user/ywkao/Outputs_sbi/pretraining/train.p"
s = load([file], with_weights=True)   # 換成你 train.p 的實際路徑
print(len(s["features"]), s["features"]["lep_pt"].shape, s["features"]["lep_pt"].dtype)
print(len(s["weights"]), s["weights"]["sm_point"].shape, s["weights"]["sm_point"].dtype)
# 預期:41  (197871,)  float64
