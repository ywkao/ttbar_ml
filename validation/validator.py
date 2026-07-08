"""
validator.py — source-agnostic comparison layer.

Only consumes two Sample objects (satisfying the schema contract); it never
knows, and shouldn't need to know, which one is nano and which is tensor.
Two orthogonal comparisons:
  - compare_features : pure kinematics, WC-independent. Histograms compared
                        after sharing binning.
  - compare_weights  : 16 operators, direct vs poly.

Comparison granularity is histogram-level, not per-event: per-event
comparison isn't possible because of random_split shuffling, batch merging,
and the absence of an event id. Histograms are immune to shuffling/merging.
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
    """Compare the histograms of two Samples feature-by-feature.

    Args:
        a, b: the two Samples to compare (convention: a=nano, b=tensor, but
            the logic is symmetric).
        names: subset of features to compare, None = all 74.
        density: True normalizes each histogram to area=1 before comparing
            (compares shape when the two populations differ in size);
            False compares raw counts (compares bin-by-bin overlap when the
            populations are the same size).

    Returns:
        {feature_name: {"max_abs_diff": float, "edges": np.ndarray,
                        "Na": np.ndarray, "Nb": np.ndarray}}
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
    """Compare direct vs poly weights per event. Both must come from the
    same nano sample and be event-aligned."""
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
