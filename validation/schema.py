"""
schema.py — the "contract layer" of the validation framework.

Defines the authoritative facts shared between the two loaders
(nanoAOD_loader, tensor_loader):
  1. FEATURE_NAMES : authoritative order of the 74 features (== the actual
                     np.concatenate column order in ml_data.calc_features:
                     29 base + 26 calc_top_features + 17 calc_pair_features).
  2. WC_NAMES      : authoritative order of SM + 16 operators.
  3. Sample        : the common schema both loaders must return.

Design principle: this file holds only "facts", no logic. It must be
importable in any environment (including machines without ROOT / torch),
so it deliberately avoids importing any heavy packages.
"""

from typing import TypedDict, Dict
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Authoritative feature order — 74 features.
# Source: the actual order produced by np.concatenate inside ml_data.py's
# calc_features(). index 0-40 are the original 41; 41-73 are additions
# (Tier 2 / Tier 1 / 3D basis / opening-angle combinations / Tier 3).
# Note: c_hel (index 55) = cos of the û_l·û_had opening angle
# (⟨c_hel⟩=−D/3), not the old cosθ_l·cosθ_had product.
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
assert len(FEATURE_NAMES) == 74, "feature count must be 74 (tensor column count)"

# name -> column index, used by tensor_loader to slice columns.
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# Authoritative WC order — 1 SM + 16 operators = 17.
# The upper-triangle packing of the 17-WC quadratic form = 17*18/2 = 153,
# exactly matching fit_coefs' column count.
# Operator order follows eft_sensitivity_scan.py's OPERATORS
# (corresponding to EFTrwgt201..216).
# ─────────────────────────────────────────────────────────────────────────────
SM_NAME = "sm_point"

OPERATORS = [
    "ctGRe", "ctGIm",
    "cQj18", "cQj38", "cQj11", "cQj31",
    "ctu8", "ctd8", "ctj8", "cQu8", "cQd8",
    "ctu1", "ctd1", "ctj1", "cQu1", "cQd1",
]
assert len(OPERATORS) == 16, "operator count must be 16"

# key set of the weights dict: SM + 16 operators.
WC_NAMES = [SM_NAME] + OPERATORS

N_FEATURES = len(FEATURE_NAMES)   # 74
N_WC = 1 + len(OPERATORS)         # 17  (including SM)
N_COEF = N_WC * (N_WC + 1) // 2   # 153 (upper-triangle packing)


# ─────────────────────────────────────────────────────────────────────────────
# Common schema — both loaders return this.
# validator only ever sees this type; it doesn't know (and shouldn't need to
# know) whether the data came from nano or tensor.
#   - features : key = one of FEATURE_NAMES, value shape = (n_events,)
#   - weights  : key = one of WC_NAMES,      value shape = (n_events,)
# nano-side weights = direct LHEWeight; tensor-side weights = reconstructed
# from fit_coefs via the polynomial fit.
# ─────────────────────────────────────────────────────────────────────────────
class Sample(TypedDict):
    features: Dict[str, np.ndarray]
    weights: Dict[str, np.ndarray]


def validate_sample(sample: "Sample", *, require_weights: bool = True) -> None:
    """Check whether a Sample satisfies the contract. Loaders can call this
    as a self-check once they're done building a Sample.

    Args:
        sample: the Sample to check.
        require_weights: set False to skip the weights check (e.g. when only
            comparing features).

    Raises:
        ValueError / KeyError: any contract violation (missing key,
            inconsistent length, etc).
    """
    feats = sample["features"]
    missing = [n for n in FEATURE_NAMES if n not in feats]
    if missing:
        raise KeyError(f"features is missing {len(missing)} authoritative keys: {missing}")

    lengths = {n: len(feats[n]) for n in FEATURE_NAMES}
    n_set = set(lengths.values())
    if len(n_set) != 1:
        raise ValueError(f"all features must have the same length, got {lengths}")

    if require_weights:
        w = sample["weights"]
        missing_w = [n for n in WC_NAMES if n not in w]
        if missing_w:
            raise KeyError(f"weights is missing keys: {missing_w}")
        n_events = n_set.pop()
        bad = {n: len(w[n]) for n in WC_NAMES if len(w[n]) != n_events}
        if bad:
            raise ValueError(f"weight length must equal feature length {n_events}, mismatch: {bad}")
