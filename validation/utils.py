"""
utils.py — plotting / 呈現工具。

把比較結果畫成圖。validator 算數字,utils 負責畫 — 兩者分開,
這樣「比對邏輯」可以在無顯示環境單獨測試,畫圖是純呈現層。
"""

from typing import Optional, Dict
import numpy as np


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
        fig.savefig(outpath, dpi=110, bbox_inches="tight")
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
    ax2.set_ylabel(r"$c_i = 1$ / SM")
    ax2.set_xlabel(name)
    ax2.set_ylim(0, 4)

    return _finish(fig, name, outpath)

