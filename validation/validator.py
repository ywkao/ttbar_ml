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
from .schema import Sample, FEATURE_NAMES, WC_NAMES


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
        names: 要比的 feature 子集,None = 全部 41 個。
        density: True 則各自正規化成面積=1 再比(母體不同時看形狀);
                 False 則比原始 count(母體相同時看逐 bin 重合)。

    Returns:
        {feature_name: {"max_abs_diff": float, "edges": np.ndarray,
                        "Na": np.ndarray, "Nb": np.ndarray}}

    實作待辦(階段 2):
        對每個 name:取共享 edges(binning.get_edges),各自 np.histogram,
        (density 時正規化),算 max|Na - Nb|,收進結果 dict。
    """
    raise NotImplementedError("階段 2 實作:共享 edges + histogram + max-diff")


def compare_weights(
    a: Sample,
    b: Sample,
    *,
    operators: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """逐 operator 比較 direct(a)vs poly(b)的 weight 分布。

    對每個 WC:回報 corr、mean±std、EFT/SM ratio 等診斷(沿用 test.py 的指標,loop 16 次)。

    實作待辦(階段 3)。
    """
    raise NotImplementedError("階段 3 實作:per-operator direct vs poly 診斷")

