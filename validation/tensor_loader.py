"""
tensor_loader.py — 從 train.p / test.p / validation.p 載入 tensor,回傳 Sample。

tensor 內容(由 ml_data.py 產出、再經 random_split):
  - features : (n_events, 72)  float64,column 順序見 schema.FEATURE_NAMES
  - fit_coefs: (n_events, 153) float64,17 個 WC 二次型的上三角 packing

職責:
  - features:用 schema.FEATURE_INDEX 把 72 個 column 切成 features dict。(直接切,不重算)
  - weights :由 fit_coefs 多項式重建 SM + 16 operator 的 weight。(階段 3 才實作)

注意:這裡的 feature 是「直接讀 tensor 既有的值」,這正是受測對象之一,
所以**不可**改用 ml_data 重算 — 那會讓 validator 失去獨立性。
"""

from typing import List, Optional
from .schema import Sample, FEATURE_INDEX, validate_sample, WC_NAMES, SM_NAME
import torch


def load(
    paths: List[str],
    *,
    with_weights: bool = False,
) -> Sample:
    """載入一個或多個 .p 檔,合併成單一 Sample。

    Args:
        paths: .p 檔路徑列表。可給多個(例如 train+test+val)以還原 split 前的全集;
               histogram 對 shuffle 免疫,所以合併順序無所謂。
        with_weights: 階段 2 設 False(只切 feature);階段 3 設 True 才重建 weight。

    Returns:
        Sample,符合 schema 契約。

    實作待辦(階段 2):
        1. torch.load 每個 path(weights_only=False),取 TensorDataset[:] 的 (feats, coefs)。
        2. .float()? — 注意 tensor 原生 float64,先保留 float64 以利精度比較,勿降精度。
        3. 沿第 0 軸 concat 多個檔。
        4. 用 FEATURE_INDEX 切 72 column 成 features dict(np.ndarray)。
        5. with_weights 時呼叫 _reconstruct_weights(coefs)(階段 3)。
    """

    feats_list, coefs_list = [], []
    for path in paths:
        _feats, _coefs = torch.load(path, weights_only=False)[:]
        feats_list.append(_feats)
        coefs_list.append(_coefs)

    feats = torch.cat(feats_list).numpy() # (n_total, 72) float64
    coefs = torch.cat(coefs_list).numpy() # (n_total, 153)

    print(feats.shape, coefs.shape)

    features = {name: feats[:, idx] for name, idx in FEATURE_INDEX.items()}
    weights = _reconstruct_weights(coefs) if with_weights else {}

    sample: Sample = {"features": features, "weights": weights}
    validate_sample(sample, require_weights=with_weights)

    return sample


def _reconstruct_weights(coefs) -> dict:
    """由 fit_coefs (n, 153) 多項式重建 SM + 16 operator 的 weight。

    對每個 WC 點 c(長度 17 的 SM-inclusive 向量),weight = coefs @ vec(outer(c, c))。
    vec 是 153 維上三角 packing,off-diagonal 的 √2 / ×2 慣例必須與 ml_data 存檔時、
    與 nano 重建時三邊一致(階段 3 開頭要先核這個慣例)。

    實作待辦(階段 3)。
    """
    import numpy as np
    from pretraining.fitCoefficients import vec, wcs as FIT_WCS
    n_wc = len(FIT_WCS) # 17

    def c_vector(op_name):
        """建一個 SM-inclusive 的 WC 向量:cSM=1,指定 operator=1,其餘 0。"""
        c = np.zeros(n_wc)
        c[FIT_WCS.index('cSM')] = 1.0          # 一律含 SM 項
        if op_name != SM_NAME:              # SM 點就只有 cSM
            c[FIT_WCS.index(op_name)] = 1.0    # 用 FIT_WCS 定位,不是 WC_NAMES!
        return c

    weights = {}
    for name in WC_NAMES:            # 直接傳 schema 名,不預先轉換
        c = c_vector(name)
        weights[name] = coefs @ vec(np.outer(c, c))

    return weights

