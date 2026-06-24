"""
utils.py — plotting / 呈現工具。

把比較結果畫成圖。validator 算數字,utils 負責畫 — 兩者分開,
這樣「比對邏輯」可以在無顯示環境單獨測試,畫圖是純呈現層。
"""

from typing import Optional
import numpy as np


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
        name: feature 名(拿來當標題 / 檔名)。
        edges: 共享 bin edges。
        Na, Nb: 兩邊的 histogram 高度(已是同一組 edges 數出來的)。
        label_a, label_b: 圖例標籤。
        outpath: 存檔路徑;None 則回傳 fig 不存。

    實作待辦(階段 2):
        上 panel:ax.stairs(Na, edges) / stairs(Nb, edges);
        下 panel:ratio = Nb/Na(避除以 0),axhline(1);存檔或回傳。
    """
    raise NotImplementedError("階段 2 實作:stairs overlay + ratio panel")

