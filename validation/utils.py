"""
utils.py — plotting / presentation helpers.

Turns comparison results into plots. validator computes the numbers, utils
does the drawing — kept separate so the comparison logic can be tested on
its own in a headless environment; plotting is a pure presentation layer.
"""

from typing import Optional, Dict
import numpy as np
import csv

from .schema import OPERATORS


def _two_panel():
    """Open a two-panel canvas (big top + small bottom, shared x-axis),
    returns (fig, ax_main, ax_ratio).

    The top panel holds the distributions, the bottom panel holds the
    ratio. Both overlay functions share this canvas, so the feature-axis
    plots (2 curves) and weight-axis plots (17 curves) look consistent.
    """
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
        sharex=True,
    )
    return fig, ax1, ax2


def _finish(fig, name, outpath):
    """Save to disk (if outpath given) or return fig (if not). Centralizes
    the wrap-up logic shared by both functions."""
    import matplotlib.pyplot as plt
    if outpath is not None:
        # fig.tight_layout() # UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
        fig.savefig(outpath, dpi=110)
        plt.close(fig)
        return outpath
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# feature axis — one feature, nano vs tensor as two curves, asking "do the
# values agree?"
# ─────────────────────────────────────────────────────────────────────────────
def overlay(
    name: str,
    edges: np.ndarray,
    Na: np.ndarray,
    Nb: np.ndarray,
    *,
    label_a: str = "nano",
    label_b: str = "tensor",
    outpath: Optional[str] = None,
):
    """Plot a single feature's overlay (two step histograms) plus a ratio
    panel below.

    Args:
        name: feature name (title / filename).
        edges: shared bin edges.
        Na, Nb: histogram heights for the two sides (already computed by
            compare_features, on the same edges).
        label_a, label_b: legend labels.
        outpath: save path; None returns fig instead.
    """
    fig, ax1, ax2 = _two_panel()

    # top: both distributions overlaid. stairs draws them as step histograms
    # using edges (the standard way to render a histogram).
    ax1.stairs(Na, edges, color="black",   linewidth=2.0, label=label_a)
    ax1.stairs(Nb, edges, color="crimson", linewidth=1.5, label=label_b, linestyle="--")
    ax1.set_ylabel("shape (normalised)")
    ax1.legend()
    ax1.set_title(name)

    # bottom: ratio = b / a. bins where Na is 0 would divide by zero, so
    # np.where fills those with nan instead.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Na > 0, Nb / Na, np.nan)
    ax2.stairs(ratio, edges, color="crimson", linewidth=1.5)
    ax2.axhline(1.0, color="grey", linewidth=1.0)
    ax2.set_ylabel(f"{label_b} / {label_a}")
    ax2.set_xlabel(name)
    ax2.set_ylim(0, 2)

    return _finish(fig, name, outpath)


# ─────────────────────────────────────────────────────────────────────────────
# weight axis — on a single feature, SM + 16 EFT operators as 17 weighted
# distributions sharing one canvas.
# ─────────────────────────────────────────────────────────────────────────────
def overlay_multi(
    name: str,
    edges: np.ndarray,
    N_sm: np.ndarray,
    N_bsm: Dict[str, np.ndarray],
    *,
    outpath: Optional[str] = None,
    ylabel_ratio: Optional[str] = r"$c_i = 1$ / SM",
):
    """Plot the overlay of SM (thick black) + 16 EFT operators (thin, colored)
    on a single feature, plus a BSM/SM ratio panel below.

    Args:
        name: feature name.
        edges: shared bin edges.
        N_sm: SM histogram heights.
        N_bsm: {operator_name: histogram heights}, 16 curves.
        outpath: save path; None returns fig instead.
    """
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = _two_panel()
    cmap = plt.colormaps.get_cmap("tab20")

    # top: 16 thin colored BSM curves, with the thick black SM curve drawn
    # on top as the reference.
    for i, (op, N) in enumerate(N_bsm.items()):
        ax1.stairs(N, edges, color=cmap(i % 20), linewidth=1.0, alpha=0.8, label=op)
    ax1.stairs(N_sm, edges, color="black", linewidth=2.0, label="SM", zorder=10)
    ax1.set_ylabel("Events (weighted)")
    ax1.legend(ncol=3, fontsize=9, loc="upper right")
    ax1.set_title(name)

    # bottom: each BSM curve's ratio to SM.
    for i, (op, N) in enumerate(N_bsm.items()):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(N_sm > 0, N / N_sm, np.nan)
        ax2.stairs(ratio, edges, color=cmap(i % 20), linewidth=1.0, alpha=0.85)
    ax2.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    ax2.set_ylabel(ylabel_ratio)
    ax2.set_xlabel(name)
    ax2.set_ylim(0, 4)

    return _finish(fig, name, outpath)


# ─────────────────────────────────────────────────────────────────────────────
# (A) validation plot: direct vs poly scatter — visualizes compare_weights'
# corr. Points all falling on the y=x diagonal means a perfect
# reconstruction; off-diagonal points are the events where it broke.
# ─────────────────────────────────────────────────────────────────────────────
def weight_scatter(
    name: str,
    w_direct: np.ndarray,
    w_poly: np.ndarray,
    *,
    corr: Optional[float] = None,
    max_n: int = 5000,
    outpath: Optional[str] = None,
):
    """One WC point: per-event scatter of direct (x) vs poly (y), plus the
    y=x diagonal.

    Args:
        name: WC name (title / filename).
        w_direct, w_poly: the two event-aligned weight arrays, same length.
        corr: if given, shown in the title (from compare_weights).
        max_n: max points to plot (randomly subsampled when there are more
            events than this, to keep the plot light).
        outpath: save path; None returns fig instead.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))

    # subsample when there are too many events; doesn't affect the trend
    n = len(w_direct)
    if n > max_n:
        idx = np.random.default_rng(0).choice(n, max_n, replace=False)
        x, y = w_direct[idx], w_poly[idx]
    else:
        x, y = w_direct, w_poly

    ax.scatter(x, y, s=4, alpha=0.3, color="steelblue")

    # y=x diagonal: points on this line mean the reconstruction is correct
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], color="crimson", linewidth=1.0, linestyle="--", label="y = x")

    ax.set_xlabel("direct LHEWeight")
    ax.set_ylabel("poly (reconstructed)")
    title = name if corr is None else f"{name}   (corr = {corr:.6f})"
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")

    return _finish(fig, name, outpath)


def normalize_hists(N_sm, N_bsm, edges):
    """Divide each by its integral (area = 1) and return the normalized versions."""
    widths = np.diff(edges)
    def norm(N):
        area = (N * widths).sum()
        return N / area if area > 0 else N
    return norm(N_sm), {op: norm(N) for op, N in N_bsm.items()}


def write_sensitivity_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["feature", "operator", "chi2_shape", "kl_div", "rate_change"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}  ({len(rows)} rows)")


def print_top_n(rows, n=20):
    print(f"\n=== Top {n} (feature, operator) by shape chi2 ===")
    rows_sorted = sorted(rows, key=lambda r: -r["chi2_shape"])
    print(f"{'feature':<12s} {'operator':<8s} {'chi2_shape':>12s} "
          f"{'KL_div':>12s} {'rate_change':>14s}")
    for r in rows_sorted[: n]:
        print(f"{r['feature']:<12s} {r['operator']:<8s} "
              f"{r['chi2_shape']:>12.4g} {r['kl_div']:>12.4g} "
              f"{r['rate_change']:>+14.3%}")


def plot_sensitivity_heatmap(rows, n=10, metric="chi2_shape", outpath=None):
    """2D heatmap: y = per-WC feature rank (1…n), x = 16 WCs.
    Color = metric / Σmetric (column-normalised per WC).
    Cell text = feature name at that rank.

    Args:
        rows:   output of validator.shape_metrics().
        n:      number of top features per WC to show (default 10).
        metric: "chi2_shape" (default), "kl_div", or "rate_change".
        outpath: save path; None returns the figure.
    """
    import matplotlib.pyplot as plt

    # {operator: {feature: value}}
    table: Dict[str, Dict[str, float]] = {}
    for r in rows:
        table.setdefault(r["operator"], {})[r["feature"]] = r[metric]

    n_ops  = len(OPERATORS)
    # z[rank, op] so operators sit on x-axis and feature rank on y-axis
    z      = np.zeros((n, n_ops))
    labels = [[""] * n_ops for _ in range(n)]

    for i, op in enumerate(OPERATORS):
        op_vals = table.get(op, {})
        total   = sum(op_vals.values())
        ranked  = sorted(op_vals, key=lambda f: -op_vals[f])[:n]
        for j, fname in enumerate(ranked):
            z[j, i]      = op_vals[fname] / total if total > 0 else 0.0
            labels[j][i] = fname

    fig, ax = plt.subplots(figsize=(n_ops * 1.2 + 1.5, n * 0.4 + 1.5))

    if metric == "rate_change":
        vmax = np.abs(z).max() or 1.0
        im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(z, aspect="auto", cmap="YlOrRd", vmin=0)

    # cell text; switch to white when background is dark
    thresh = z.max() * 0.6
    fsize  = max(5, 8 - n // 20)   # shrink font for large n
    for j in range(n):
        for i in range(n_ops):
            color = "white" if z[j, i] > thresh else "black"
            ax.text(i, j, labels[j][i],
                    ha="center", va="center", fontsize=fsize, color=color)

    ax.set_xticks(range(n_ops))
    ax.set_xticklabels(OPERATORS, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"#{j+1}" for j in range(n)], fontsize=9)
    ax.invert_yaxis()
    ax.set_ylabel("Feature rank (per operator)")
    ax.set_title(f"Per-operator feature sensitivity   {metric} / Σ{metric}")
    plt.colorbar(im, ax=ax, label=f"{metric} / Σ{metric}", fraction=0.02, pad=0.02)
    fig.tight_layout()

    return _finish(fig, "sensitivity_heatmap", outpath)


def print_best_per_op(rows):
    print(f"\n=== Best feature for each operator (by shape chi2) ===")
    print(f"{'operator':<8s} {'best feature':<12s} {'chi2_shape':>12s} {'KL_div':>12s}")
    for op in OPERATORS:
        best = max((r for r in rows if r["operator"] == op),
                   key=lambda r: r["chi2_shape"])
        print(f"{op:<8s} {best['feature']:<12s} "
              f"{best['chi2_shape']:>12.4g} {best['kl_div']:>12.4g}")


def _fmt_sci(val: float, latex: bool = False) -> str:
    """Format val as 2-significant-figure scientific notation, plain or LaTeX."""
    s = f"{val:.1e}"          # e.g. "1.2e-04"
    if not latex:
        return s
    mantissa, exp_str = s.split("e")
    exp = int(exp_str)
    return rf"${mantissa} \times 10^{{{exp}}}$"


def print_feature_wc_table(rows, n=10, metric="chi2_shape", latex=False, transpose=True):
    """Print a sensitivity table: top-N features (ranked by total metric) vs 16 WCs.

    Args:
        rows:      output of validator.shape_metrics().
        n:         number of top features (default 10).
        metric:    cell value — "chi2_shape" (default), "kl_div", or "rate_change".
        latex:     if True, emit a LaTeX tabular environment.
        transpose: if True (default), rows = 16 WCs, columns = top-N features.
                   if False, rows = top-N features, columns = 16 WCs.
    """
    table: Dict[str, Dict[str, float]] = {}
    for r in rows:
        table.setdefault(r["feature"], {})[r["operator"]] = r[metric]

    feature_score = {f: sum(vals.values()) for f, vals in table.items()}
    top_features = sorted(feature_score, key=lambda f: -feature_score[f])[:n]

    print(f"\n=== Top {n} features by total {metric} across 16 WCs ===")

    # pre-escape feature names for LaTeX (backslash not allowed inside f-strings)
    feat_tex = [f.replace("_", r"\_") for f in top_features]

    if latex:
        if transpose:
            # rows = WCs, columns = top-N features
            col_spec = "l" + "r" * len(top_features)
            header_cells = [r"\textbf{Operator}"] + [
                r"\textbf{" + t + "}" for t in feat_tex
            ]
            print(rf"\begin{{tabular}}{{{col_spec}}}")
            print(r"\hline")
            print(" & ".join(header_cells) + r" \\")
            print(r"\hline")
            for op in OPERATORS:
                cells = [op] + [_fmt_sci(table[f].get(op, 0.0), latex=True) for f in top_features]
                print(" & ".join(cells) + r" \\")
            print(r"\hline")
            total_cells = [r"\textbf{$\Sigma$(chi2)}"] + [
                _fmt_sci(feature_score[f], latex=True) for f in top_features
            ]
            print(" & ".join(total_cells) + r" \\")
            print(r"\hline")
            print(r"\end{tabular}")
        else:
            # rows = top-N features, columns = WCs
            col_spec = "l" + "r" * (len(OPERATORS) + 1)
            header_cells = (
                [r"\textbf{Feature}"]
                + [rf"\textbf{{{op}}}" for op in OPERATORS]
                + [r"\textbf{$\Sigma$(chi2)}"]
            )
            print(rf"\begin{{tabular}}{{{col_spec}}}")
            print(r"\hline")
            print(" & ".join(header_cells) + r" \\")
            print(r"\hline")
            for fname, fname_tex in zip(top_features, feat_tex):
                cells = (
                    [fname_tex]
                    + [_fmt_sci(table[fname].get(op, 0.0), latex=True) for op in OPERATORS]
                    + [_fmt_sci(feature_score[fname], latex=True)]
                )
                print(" & ".join(cells) + r" \\")
            print(r"\hline")
            print(r"\end{tabular}")
    else:
        if transpose:
            # rows = WCs, columns = top-N features
            op_w  = max(len("Operator"), max(len(op) for op in OPERATORS) )
            col_w = max(9, max(len(f) for f in top_features)) + 1

            header = f"{'Operator':<{op_w}s}" + "".join(f"{f:>{col_w}s}" for f in top_features)
            sep = "-" * len(header)

            print(header)
            print(sep)
            for op in OPERATORS:
                vals_row = "".join(
                    f"{_fmt_sci(table[f].get(op, 0.0)):>{col_w}s}" for f in top_features
                )
                print(f"{op:<{op_w}s}{vals_row}")
            print(sep)
            total_row = "".join(f"{_fmt_sci(feature_score[f]):>{col_w}s}" for f in top_features)
            print(f"{'Σ(chi2)':<{op_w}s}{total_row}")
            print()
        else:
            # rows = top-N features, columns = WCs
            col_w  = 10
            feat_w = 16
            header = (
                f"{'Rank':<4s} {'Feature':<{feat_w}s}"
                + "".join(f"{op:>{col_w}s}" for op in OPERATORS)
                + f"{'Σ(chi2)':>{col_w}s}"
            )
            sep = "-" * len(header)

            print(header)
            print(sep)
            for rank, fname in enumerate(top_features, 1):
                cells = "".join(
                    f"{_fmt_sci(table[fname].get(op, 0.0)):>{col_w}s}" for op in OPERATORS
                )
                print(f"{rank:<4d} {fname:<{feat_w}s}{cells}{_fmt_sci(feature_score[fname]):>{col_w}s}")
            print()
