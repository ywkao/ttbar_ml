"""
validator.py — source-agnostic 比較層。

只吃兩個 Sample(符合 schema 契約),不知道也不該知道誰是 nano、誰是 tensor。
兩個正交的比較:
  - compare_features : 純運動學,WC-independent。共享 binning 後比 histogram。
  - compare_weights  : 16 operator,direct vs poly。(階段 3)

比較強度:histogram-level(階段 0 審計結論 — per-event 因 random_split shuffle +
batch merge + 無 event id 而做不到)。histogram 對 shuffle/merge 免疫。
"""

from typing import Dict, List, Optional
import numpy as np
from .schema import Sample, FEATURE_NAMES, WC_NAMES, OPERATORS, SM_NAME
from .binning import get_edges


def compare_features(
    a: Sample,
    b: Sample,
    *,
    names: Optional[List[str]] = None,
    density: bool = True,
) -> Dict[str, dict]:
    """逐 feature 比較兩個 Sample 的 histogram。

    Args:
        a, b: 兩個待比 Sample(慣例:a=nano, b=tensor,但邏輯對稱)。
        names: 要比的 feature 子集,None = 全部 72 個。
        density: True 則各自正規化成面積=1 再比(母體不同時看形狀);
                 False 則比原始 count(母體相同時看逐 bin 重合)。

    Returns:
        {feature_name: {"max_abs_diff": float, "edges": np.ndarray,
                        "Na": np.ndarray, "Nb": np.ndarray}}

    實作待辦(階段 2):
        對每個 name:取共享 edges(binning.get_edges),各自 np.histogram,
        (density 時正規化),算 max|Na - Nb|,收進結果 dict。
    """

    if names is None:
        names = FEATURE_NAMES

    results = {}
    for name in names:
        edges = get_edges(name)
        Na, _ = np.histogram(a["features"][name], bins=edges, density=density)
        Nb, _ = np.histogram(b["features"][name], bins=edges, density=density)
        results[name] = {
            "edges": edges,
            "Na": Na,
            "Nb": Nb,
            "max_abs_diff": float(np.max(np.abs(Na-Nb))),
        }

    return results


def compare_weights(direct: dict, poly: dict, *, operators=None) -> dict:
    """per-event 比 direct vs poly。兩者須來自同一份 nano、逐 event 對齊。"""
    if operators is None:
        operators = WC_NAMES
    results = {}
    for name in operators:
        wd, wp = direct[name], poly[name]
        results[name] = {
            "corr": float(np.corrcoef(wd, wp)[0, 1]),
            "max_abs_diff": float(np.max(np.abs(wd - wp))),
            "mean_direct": float(wd.mean()), "mean_poly": float(wp.mean()),
        }
    return results


def _shape_chi2(N_bsm, N_sm):
    """Pearson chi2 between shape-normalized PMFs."""
    s_bsm, s_sm = N_bsm.sum(), N_sm.sum()
    if s_bsm <= 0 or s_sm <= 0:
        return 0.0
    p_bsm = N_bsm / s_bsm
    p_sm  = N_sm  / s_sm
    mask = p_sm > 0
    return float(np.sum((p_bsm[mask] - p_sm[mask]) ** 2 / p_sm[mask]))

def _kl_divergence(N_bsm, N_sm):
    """KL(BSM || SM) on shape-normalized PMFs."""
    s_bsm, s_sm = N_bsm.sum(), N_sm.sum()
    if s_bsm <= 0 or s_sm <= 0:
        return 0.0
    p_bsm = N_bsm / s_bsm
    p_sm  = N_sm  / s_sm
    mask = (p_bsm > 0) & (p_sm > 0)
    return float(np.sum(p_bsm[mask] * np.log(p_bsm[mask] / p_sm[mask])))

def _rate_change(w_bsm, w_sm):
    s_sm = w_sm.sum()
    if s_sm == 0:
        return 0.0
    return float((w_bsm.sum() - s_sm) / s_sm)

def shape_metrics(sample, *, operators=None):
    if operators is None:
        operators = OPERATORS
    rows = []
    for fname in FEATURE_NAMES:
        edges = get_edges(fname)
        vals = sample["features"][fname]
        N_sm = np.histogram(vals, bins=edges, weights=sample["weights"][SM_NAME])[0]
        for op in operators:
            N_op = np.histogram(vals, bins=edges, weights=sample["weights"][op])[0]
            rows.append({
                "feature": fname, "operator": op,
                "chi2_shape": _shape_chi2(N_op, N_sm),
                "kl_div":     _kl_divergence(N_op, N_sm),
                "rate_change": _rate_change(N_op, N_sm),
            })
    return rows
