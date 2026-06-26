"""
schema.py — 驗證框架的「契約層」。

這個檔案定義兩個 loader (nanoAOD_loader, tensor_loader) 之間共用的權威事實:
  1. FEATURE_NAMES : 74 個 feature 的權威順序(== ml_data.calc_features 的
                     np.concatenate 實際 column 順序:29 base + 26 calc_top_features
                     + 17 calc_pair_features)。
  2. WC_NAMES      : SM + 16 operator 的權威順序。
  3. Sample        : 兩個 loader 都必須回傳的共同 schema。

設計原則:這裡只放「事實」,不放邏輯。任何環境(含沒有 ROOT / torch 的機器)
都能 import 它,所以這裡刻意不 import 任何重套件。
"""

from typing import TypedDict, Dict
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 權威 feature 順序 — 74 個。
# 來源:ml_data.py 的 calc_features() 內 np.concatenate 的實際次序。
# index 0-40 為原始 41 個;41-73 為新增(Tier 2 / Tier 1 / 3D 基底 / 開角組合 / Tier 3)。
# 註:c_hel(index 55)= û_l·û_had 開角餘弦(⟨c_hel⟩=−D/3),非舊版 cosθ_l·cosθ_had 乘積。
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    "lep_pt", "lep_eta", "lep_phi", "lep_mass",
    "met_pt", "met_phi",
    "jet0_pt", "jet0_eta", "jet0_phi", "jet0_mass",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_mass",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_mass",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_mass",
    "njets", "HT", "mT_W",
    "jet0_flav", "jet1_flav", "jet2_flav", "jet3_flav",
    "lep_top_pt", "lep_top_eta", "lep_top_phi", "lep_top_mass",
    "had_top_pt", "had_top_eta", "had_top_phi", "had_top_mass",
    "dr_tt", "dr_lep_had", "m_ttbar", "cos_theta_star",
    # ── Tier 2: tt̄ system kinematics ──
    "dy_tt", "dphi_tt", "pt_tt", "y_tt",
    # ── Tier 1: spin correlation / polarization (helicity frame) ──
    "cos_theta_l", "cos_theta_had", "dphi_l_had",
    # ── 3D spin-basis projections (common {n,r,k}) ──
    "cos_lep_n", "cos_lep_r", "cos_lep_k",
    "cos_had_n", "cos_had_r", "cos_had_k",
    # ── opening-angle / relative-velocity combinations ──
    "beta_t_star", "c_hel", "c_han",
    # ── Tier 3: pairwise lepton-jet / jet-jet geometry & masses ──
    "dr_l_j0", "dr_l_j1", "dr_l_j2", "dr_l_j3",
    "dr_j01", "dr_j02", "dr_j03", "dr_j12", "dr_j13", "dr_j23",
    "m_j01", "m_j02", "m_j03", "m_j12", "m_j13", "m_j23",
    "m_lb_min",
]
assert len(FEATURE_NAMES) == 74, "feature 數必須是 74(tensor column 數)"

# name -> column index,給 tensor_loader 切 column 用。
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# 權威 WC 順序 — 1 個 SM + 16 operator = 17。
# 17 個 WC 的二次型上三角 = 17*18/2 = 153,正好等於 fit_coefs 的 column 數。
# operator 順序沿用 eft_sensitivity_scan.py 的 OPERATORS(對應 EFTrwgt201..216)。
# ─────────────────────────────────────────────────────────────────────────────
SM_NAME = "sm_point"

OPERATORS = [
    "ctGRe", "ctGIm",
    "cQj18", "cQj38", "cQj11", "cQj31",
    "ctu8", "ctd8", "ctj8", "cQu8", "cQd8",
    "ctu1", "ctd1", "ctj1", "cQu1", "cQd1",
]
assert len(OPERATORS) == 16, "operator 數必須是 16"

# weights dict 的 key 集合:SM + 16 operator。
WC_NAMES = [SM_NAME] + OPERATORS

N_FEATURES = len(FEATURE_NAMES)   # 74
N_WC = 1 + len(OPERATORS)         # 17  (含 SM)
N_COEF = N_WC * (N_WC + 1) // 2   # 153 (上三角 packing)


# ─────────────────────────────────────────────────────────────────────────────
# 共同 schema — 兩個 loader 都回傳這個。
# validator 只看到這個型別,不知道(也不該知道)資料來自 nano 還是 tensor。
#   - features : key = FEATURE_NAMES 之一, value shape = (n_events,)
#   - weights  : key = WC_NAMES 之一,      value shape = (n_events,)
# nano 側 weights = direct LHEWeight;tensor 側 weights = 由 fit_coefs 多項式重建。
# ─────────────────────────────────────────────────────────────────────────────
class Sample(TypedDict):
    features: Dict[str, np.ndarray]
    weights: Dict[str, np.ndarray]


def validate_sample(sample: "Sample", *, require_weights: bool = True) -> None:
    """檢查一個 Sample 是否符合契約。loader 寫完後可呼叫這個自我檢查。

    Args:
        sample: 待檢查的 Sample。
        require_weights: 階段 2(只比 feature)時可設 False,跳過 weights 檢查。

    Raises:
        ValueError / KeyError: 任何與契約不符之處(缺 key、長度不一致等)。
    """
    feats = sample["features"]
    missing = [n for n in FEATURE_NAMES if n not in feats]
    if missing:
        raise KeyError(f"features 缺少 {len(missing)} 個權威 key: {missing}")

    lengths = {n: len(feats[n]) for n in FEATURE_NAMES}
    n_set = set(lengths.values())
    if len(n_set) != 1:
        raise ValueError(f"所有 feature 長度必須一致,實得 {lengths}")

    if require_weights:
        w = sample["weights"]
        missing_w = [n for n in WC_NAMES if n not in w]
        if missing_w:
            raise KeyError(f"weights 缺少 key: {missing_w}")
        n_events = n_set.pop()
        bad = {n: len(w[n]) for n in WC_NAMES if len(w[n]) != n_events}
        if bad:
            raise ValueError(f"weight 長度必須等於 feature 長度 {n_events},不符: {bad}")

