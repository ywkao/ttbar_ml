"""
utils.py — plotting / 呈現工具。

把比較結果畫成圖。validator 算數字,utils 負責畫 — 兩者分開,
這樣「比對邏輯」可以在無顯示環境單獨測試,畫圖是純呈現層。
"""

from typing import Optional, Dict
import numpy as np
import csv

from .schema import OPERATORS


def _two_panel():
    """開一個「上大下小 + 共用 x 軸」的雙 panel 畫布,回傳 (fig, ax_main, ax_ratio)。

    上 panel 放分布、下 panel 放 ratio。兩個 overlay 函式共用這個畫布,
    所以 feature 軸(2 條)跟 weight 軸(17 條)畫出來長相一致。
    """
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05},
        sharex=True,
    )
    return fig, ax1, ax2


def _finish(fig, name, outpath):
    """存檔(有 outpath)或回傳 fig(無)。集中處理收尾,兩個函式共用。"""
    import matplotlib.pyplot as plt
    if outpath is not None:
        # fig.tight_layout() # UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
        fig.savefig(outpath, dpi=110)
        plt.close(fig)
        return outpath
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 階段 2:feature 軸 — 同一個 feature,nano vs tensor 兩條,問「值算得一致嗎」。
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
    """畫單一 feature 的 overlay(兩條 step histogram)+ 下方 ratio panel。

    Args:
        name: feature 名(標題 / 檔名)。
        edges: 共享 bin edges。
        Na, Nb: 兩邊的 histogram 高度(compare_features 已算好,同一組 edges)。
        label_a, label_b: 圖例標籤。
        outpath: 存檔路徑;None 則回傳 fig。
    """
    fig, ax1, ax2 = _two_panel()

    # 上:兩條分布疊一起。stairs 用 edges 畫成階梯狀(histogram 的標準畫法)。
    ax1.stairs(Na, edges, color="black",   linewidth=2.0, label=label_a)
    ax1.stairs(Nb, edges, color="crimson", linewidth=1.5, label=label_b, linestyle="--")
    ax1.set_ylabel("shape (normalised)")
    ax1.legend()
    ax1.set_title(name)

    # 下:ratio = b / a。Na 為 0 的 bin 會除以 0,用 np.where 填 nan 跳過。
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(Na > 0, Nb / Na, np.nan)
    ax2.stairs(ratio, edges, color="crimson", linewidth=1.5)
    ax2.axhline(1.0, color="grey", linewidth=1.0)
    ax2.set_ylabel(f"{label_b} / {label_a}")
    ax2.set_xlabel(name)
    ax2.set_ylim(0, 2)

    return _finish(fig, name, outpath)


# ─────────────────────────────────────────────────────────────────────────────
# 階段 3 預備:weight 軸 — 同一個 feature 上 SM + 16 EFT 共 17 條 weighted 分布。
# 這就是你心裡那張圖。資料來源不同(17 條 weighted hist),畫布同一個。
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
    """畫單一 feature 上 SM(粗黑)+ 16 EFT(細彩)的 overlay + 下方 BSM/SM ratio。

    Args:
        name: feature 名。
        edges: 共享 bin edges。
        N_sm: SM 的 histogram 高度。
        N_bsm: {operator_name: histogram 高度},共 16 條。
        outpath: 存檔路徑;None 則回傳 fig。

    註:這個函式階段 3 才會被餵真實資料(需要先做 weight 重建)。
        現在先放著,當作你那張 EFT 圖的畫布。
    """
    import matplotlib.pyplot as plt
    fig, ax1, ax2 = _two_panel()
    cmap = plt.colormaps.get_cmap("tab20")

    # 上:16 條 BSM 細線(彩),SM 粗黑線壓在最上面當基準。
    for i, (op, N) in enumerate(N_bsm.items()):
        ax1.stairs(N, edges, color=cmap(i % 20), linewidth=1.0, alpha=0.8, label=op)
    ax1.stairs(N_sm, edges, color="black", linewidth=2.0, label="SM", zorder=10)
    ax1.set_ylabel("Events (weighted)")
    ax1.legend(ncol=3, fontsize=9, loc="upper right")
    ax1.set_title(name)

    # 下:每條 BSM 對 SM 的 ratio。
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
# (A) 驗證圖:direct vs poly 的 scatter — 把 compare_weights 的 corr 視覺化。
# 點全落在 y=x 對角線 → 重建完美。離線的點就是重建出問題的 event。
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
    """單一 WC 點:direct(x)vs poly(y)的逐 event scatter + y=x 對角線。

    Args:
        name: WC 名(標題 / 檔名)。
        w_direct, w_poly: 逐 event 對齊的兩套 weight,長度相同。
        corr: 若給,標在標題上(從 compare_weights 拿)。
        max_n: 最多畫幾個點(event 太多時隨機抽樣,免得圖太重)。
        outpath: 存檔路徑;None 則回傳 fig。
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))

    # event 太多就抽樣畫,不影響趨勢判讀
    n = len(w_direct)
    if n > max_n:
        idx = np.random.default_rng(0).choice(n, max_n, replace=False)
        x, y = w_direct[idx], w_poly[idx]
    else:
        x, y = w_direct, w_poly

    ax.scatter(x, y, s=4, alpha=0.3, color="steelblue")

    # y=x 對角線:點貼這條線 = 重建正確
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
    """各自除以積分(面積=1),回傳 normalized 版。"""
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
    """2D heatmap: y = 16 WCs, x = per-WC feature rank (1…n).
    Color = metric / Σmetric (row-normalised per WC).
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
    z      = np.zeros((n_ops, n))
    labels = [[""] * n for _ in range(n_ops)]

    for i, op in enumerate(OPERATORS):
        op_vals = table.get(op, {})
        total   = sum(op_vals.values())
        ranked  = sorted(op_vals, key=lambda f: -op_vals[f])[:n]
        for j, fname in enumerate(ranked):
            z[i, j]      = op_vals[fname] / total if total > 0 else 0.0
            labels[i][j] = fname

    fig, ax = plt.subplots(figsize=(n * 1.8 + 1.5, n_ops * 0.65 + 1.2))

    if metric == "rate_change":
        vmax = np.abs(z).max() or 1.0
        im = ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(z, aspect="auto", cmap="YlOrRd", vmin=0)

    # cell text; switch to white when background is dark
    thresh = z.max() * 0.6
    for i in range(n_ops):
        for j in range(n):
            color = "white" if z[i, j] > thresh else "black"
            ax.text(j, i, labels[i][j],
                    ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels([f"#{j+1}" for j in range(n)], fontsize=9)
    ax.set_yticks(range(n_ops))
    ax.set_yticklabels(OPERATORS, fontsize=9)
    ax.set_xlabel("Feature rank (per operator)")
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
