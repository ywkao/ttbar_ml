"""
nanoAOD_loader.py — 從 NanoAOD root 檔獨立重算 features / weights,回傳 Sample。

職責:
  - selection:import 共享的 analysis_tools(genObjectSelection / genEventSelection),
               **不可**抄 eft_sensitivity_scan.py 裡 local 那份 — selection 不一致是 noise,
               必須與 ml_data 用同一份。
  - features :獨立重算 41 個 feature(這是受測對象,必須與 tensor 側互不依賴)。
               完整 41 個的計算邏輯參考 test.py 的 inline 版本(eft_scan 少 4 個,不可用)。
  - weights  :對 SM + 16 operator 取 direct LHEWeight(階段 3)。
               operator -> rwgt 欄位的對應參考 eft_sensitivity_scan.find_linear_rwgts。

設計選擇(階段 0 batch-mode 審計的結論):
  train.p 是「多個 root merge 後再 random_split」的子集,且無 event id。
  所以單一 root 的母體 != train.p 的母體,兩邊 histogram 只會「形狀相似」。
  要嚴格重合,需載入與產生 train.p 同一批 root,並比 split 前全集。
  介面因此設計成吃 List[str](多檔)。
"""

from typing import List, Optional
from .schema import Sample


def load(
    files: List[str],
    *,
    nevents: Optional[int] = None,
    with_weights: bool = False,
) -> Sample:
    """載入一個或多個 root 檔,回傳 Sample。

    Args:
        files: NanoAOD root 檔路徑列表。
        nevents: 每檔最多取前 N 個 event(debug 用,None = 全部)。
        with_weights: 階段 2 設 False;階段 3 設 True 才取 direct LHEWeight。

    Returns:
        Sample,符合 schema 契約。

    實作待辦(階段 2):
        1. NanoEventsFactory.from_root 載入,選擇性截 nevents。
        2. 共享 selection:cleanleps, cleanjets = genObjectSelection(events);
           mask = genEventSelection(...)。
        3. 重算 41 個 feature(參考 test.py inline)成 features dict(np.ndarray)。
        4. with_weights 時呼叫 _direct_weights(events, mask)(階段 3)。
        5. 多檔則各自處理後沿第 0 軸 concat。
    """
    raise NotImplementedError("階段 2 實作:共享 selection + 獨立重算 41 feature")


def _direct_weights(events, mask) -> dict:
    """對 SM + 16 operator 取 direct LHEWeight,回傳 weights dict。

    - SM    : events.LHEWeight["sm_point"][mask]
    - op_i  : events.LHEWeight[rwgt_field_i][mask],rwgt_field 由 find_linear_rwgts 解析。

    實作待辦(階段 3)。
    """
    raise NotImplementedError("階段 3 實作:direct LHEWeight 取 16+1 個 weight")

