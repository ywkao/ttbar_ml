"""
binning.py — 共享的 bin edges 設定(契約層)。

為什麼要獨立一個檔案、兩個 loader 共用:
  histogram-level 比較成立的前提是「兩邊用同一組 bin edges」。
  ipynb 的做法是 np.linspace(vals.min(), vals.max(), ...) — range 依賴資料本身,
  nano 與 tensor 的 min/max 必然不同,bin edges 就不同,histogram 天生對不齊。
  所以 range 必須寫死成共享常數,不能從資料推。

涵蓋全部 41 個 feature(數字主要沿用 eft_sensitivity_scan.py 的 BINNING,
結尾四個 dr_tt/dr_lep_had/m_ttbar/cos_theta_star 沿用 test.py 的 BINNING)。
"""

import numpy as np
from .schema import FEATURE_NAMES


# 每個 feature: (nbins, lo, hi)
BINNING = {
    "lep_pt":   (50,  0,    500),
    "lep_eta":  (50, -3,      3),
    "lep_phi":  (40, -np.pi,  np.pi),
    "lep_mass": (50,  0,      0.2),
    "met_pt":   (50,  0,    500),
    "met_phi":  (40, -np.pi,  np.pi),
    "njets":    (10,  3.5,  13.5),
    "HT":       (60,  0,   2500),
    "mT_W":     (50,  0,    350),
}
for _i in range(4):
    BINNING[f"jet{_i}_pt"]   = (50, 0, 700 if _i == 0 else 500)
    BINNING[f"jet{_i}_eta"]  = (50, -3, 3)
    BINNING[f"jet{_i}_phi"]  = (40, -np.pi, np.pi)
    BINNING[f"jet{_i}_mass"] = (50, 0, 60)
    BINNING[f"jet{_i}_flav"] = (6, -0.5, 5.5)
for _top in ("lep_top", "had_top"):
    BINNING[f"{_top}_pt"]   = (50,  0,   800)
    BINNING[f"{_top}_eta"]  = (50, -4,     4)
    BINNING[f"{_top}_phi"]  = (40, -np.pi, np.pi)
    BINNING[f"{_top}_mass"] = (60, 100,  250)

BINNING["dr_tt"]          = (40,  0,  6)
BINNING["dr_lep_had"]     = (40,  0,  6)
BINNING["m_ttbar"]        = (40,  0, 2500)
BINNING["cos_theta_star"] = (40, -1,  1)

# 契約自檢:binning 必須剛好覆蓋 41 個權威 feature,不多不少。
_missing = [n for n in FEATURE_NAMES if n not in BINNING]
_extra = [n for n in BINNING if n not in FEATURE_NAMES]
assert not _missing, f"binning 缺少 feature: {_missing}"
assert not _extra, f"binning 有多餘的 key(不在權威表內): {_extra}"


def get_edges(name: str) -> np.ndarray:
    """回傳某 feature 的 bin edges 陣列,長度 = nbins + 1。"""
    nb, lo, hi = BINNING[name]
    return np.linspace(lo, hi, nb + 1)

