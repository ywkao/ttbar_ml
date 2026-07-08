"""
tensor_loader.py — loads a tensor from train.p / test.p / validation.p and
returns a Sample.

Tensor contents (produced by ml_data.py, then passed through random_split):
  - features : (n_events, 74)  float64, column order per schema.FEATURE_NAMES
  - fit_coefs: (n_events, 153) float64, upper-triangle packing of the
               17-WC quadratic form

Responsibilities:
  - features: slice the 74 columns into a features dict using
    schema.FEATURE_INDEX (a direct slice, no recomputation).
  - weights : reconstruct SM + 16 operator weights from fit_coefs via the
    polynomial fit.

Note: the features here are read directly from the tensor's existing
values — that's exactly what's under test, so this must **not** be swapped
for a recompute via ml_data, which would make the validator lose its
independence.
"""

from typing import List, Optional
from .schema import Sample, FEATURE_INDEX, validate_sample, WC_NAMES, SM_NAME
import torch


def load(
    paths: List[str],
    *,
    with_weights: bool = False,
    wc_value: float = 1.0,
) -> Sample:
    """Load one or more .p files, merged into a single Sample.

    Args:
        paths: list of .p file paths. Multiple paths (e.g. train+test+val)
            can be given to reconstruct the full set prior to the split;
            histograms are immune to shuffling, so merge order doesn't
            matter.
        with_weights: False to only slice features; True to also
            reconstruct weights.
        wc_value: the coefficient assigned to the operator when
            reconstructing BSM weights (the SM term is always cSM=1).
            Note this only affects the tensor-side reconstruction and
            doesn't correspond to any actual LHE reweight point — it's
            purely for inspecting shape/sensitivity at different WC scales.

    Returns:
        A Sample satisfying the schema contract.
    """

    feats_list, coefs_list = [], []
    for path in paths:
        _feats, _coefs = torch.load(path, weights_only=False)[:]
        feats_list.append(_feats)
        coefs_list.append(_coefs)

    feats = torch.cat(feats_list).numpy() # (n_total, 74) float64
    coefs = torch.cat(coefs_list).numpy() # (n_total, 153)

    print(feats.shape, coefs.shape)

    features = {name: feats[:, idx] for name, idx in FEATURE_INDEX.items()}
    weights = _reconstruct_weights(coefs, wc_value=wc_value) if with_weights else {}

    sample: Sample = {"features": features, "weights": weights}
    validate_sample(sample, require_weights=with_weights)

    return sample


def _reconstruct_weights(coefs, *, wc_value: float = 1.0) -> dict:
    """Reconstruct SM + 16 operator weights from fit_coefs (n, 153) via the
    polynomial fit.

    For each WC point c (a length-17 SM-inclusive vector),
    weight = coefs @ vec(outer(c, c)). vec is the 153-dim upper-triangle
    packing; the off-diagonal √2 / ×2 convention must match across all
    three sides — how ml_data saves it, and how the nano-side reconstruction
    does it.

    Args:
        wc_value: the coefficient assigned to the operator (the SM term is
            always cSM=1). wc_value=1.0 matches the nano-side direct/poly
            comparison point; any other value is purely a tensor-side
            extrapolation view with no corresponding LHE reweight point.
    """
    import numpy as np
    from pretraining.fitCoefficients import vec, wcs as FIT_WCS
    n_wc = len(FIT_WCS) # 17

    def c_vector(op_name):
        """Build an SM-inclusive WC vector: cSM=1, the given operator=wc_value, rest 0."""
        c = np.zeros(n_wc)
        c[FIT_WCS.index('cSM')] = 1.0          # always include the SM term
        if op_name != SM_NAME:              # the SM point only has cSM
            c[FIT_WCS.index(op_name)] = wc_value  # indexed via FIT_WCS, not WC_NAMES!
        return c

    weights = {}
    for name in WC_NAMES:            # pass schema names through directly, no pre-conversion
        c = c_vector(name)
        weights[name] = coefs @ vec(np.outer(c, c))

    return weights
