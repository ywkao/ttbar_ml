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
from .schema import Sample, FEATURE_NAMES, validate_sample, SM_NAME, OPERATORS, WC_NAMES
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from pretraining.analysis_tools import genObjectSelection, genEventSelection

import numpy as np
import awkward as ak
import re

NanoAODSchema.warn_missing_crossrefs = False


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

    # FILE = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
    # N_EVENTS = 2000   # increase for more statistics
    # OUTDIR   = "plots_validation"

    per_file, per_file_w = [], []
    for f in files:
        print(f"Loading {nevents} events from {f}")
        events = NanoEventsFactory.from_root(
            {f: 'Events'}, schemaclass=NanoAODSchema
        ).events()[:nevents]

        cleanleps, cleanjets = genObjectSelection(events)
        mask = genEventSelection(cleanleps, cleanjets)
        print(f"Selected {int(ak.sum(mask))} / {len(events)} events")

        leps = ak.flatten(cleanleps[mask])
        jets = cleanjets[mask]
        met  = events.GenMET[mask]

        gp = events.GenPart[mask]
        is_final = gp.hasFlags(["fromHardProcess", "isLastCopy"])
        t_gp    = ak.pad_none(gp[is_final & (gp.pdgId ==  6)], 1)[:, 0]
        tbar_gp = ak.pad_none(gp[is_final & (gp.pdgId == -6)], 1)[:, 0]

        def to_np(arr):
            return ak.to_numpy(ak.fill_none(arr, 0.)).astype(float)

        t_pt   = to_np(t_gp.pt);   t_eta  = to_np(t_gp.eta)
        t_phi  = to_np(t_gp.phi);  t_mass = to_np(t_gp.mass)
        tb_pt  = to_np(tbar_gp.pt);  tb_eta = to_np(tbar_gp.eta)
        tb_phi = to_np(tbar_gp.phi); tb_mass = to_np(tbar_gp.mass)

        t_px  = t_pt  * np.cos(t_phi);   t_py  = t_pt  * np.sin(t_phi)
        t_pz  = t_pt  * np.sinh(t_eta);  t_e   = np.sqrt(t_px**2 + t_py**2 + t_pz**2 + t_mass**2)
        tb_px = tb_pt * np.cos(tb_phi);  tb_py = tb_pt * np.sin(tb_phi)
        tb_pz = tb_pt * np.sinh(tb_eta); tb_e  = np.sqrt(tb_px**2 + tb_py**2 + tb_pz**2 + tb_mass**2)
        sys_px = t_px + tb_px;  sys_py = t_py + tb_py
        sys_pz = t_pz + tb_pz;  sys_e  = t_e  + tb_e

        lep_phi_np = ak.to_numpy(leps.phi).astype(float)
        met_pt_np  = ak.to_numpy(met.pt).astype(float)
        met_phi_np = ak.to_numpy(met.phi).astype(float)
        mT_W = np.sqrt(2 * ak.to_numpy(leps.pt).astype(float) * met_pt_np
                       * (1 - np.cos(lep_phi_np - met_phi_np)))

        m_ttbar = np.sqrt(np.maximum(sys_e**2 - sys_px**2 - sys_py**2 - sys_pz**2, 0.))

        deta_tt = t_eta - tb_eta
        dphi_tt = np.arctan2(np.sin(t_phi - tb_phi), np.cos(t_phi - tb_phi))
        dr_tt   = np.sqrt(deta_tt**2 + dphi_tt**2)

        lep_is_neg  = ak.to_numpy(leps.pdgId > 0)
        had_top_eta = np.where(lep_is_neg, t_eta,  tb_eta)
        had_top_phi = np.where(lep_is_neg, t_phi,  tb_phi)
        lep_eta_np  = ak.to_numpy(leps.eta).astype(float)
        deta_lh    = lep_eta_np - had_top_eta
        dphi_lh    = np.arctan2(np.sin(lep_phi_np - had_top_phi), np.cos(lep_phi_np - had_top_phi))
        dr_lep_had = np.sqrt(deta_lh**2 + dphi_lh**2)

        beta2  = (sys_px**2 + sys_py**2 + sys_pz**2) / np.maximum(sys_e**2, 1e-10)
        gamma  = 1. / np.sqrt(np.maximum(1. - beta2, 1e-10))
        bx = sys_px / np.maximum(sys_e, 1e-10)
        by = sys_py / np.maximum(sys_e, 1e-10)
        bz = sys_pz / np.maximum(sys_e, 1e-10)
        bp = bx*t_px + by*t_py + bz*t_pz
        gamma2     = np.where(beta2 > 1e-10, (gamma - 1.) / beta2, 0.)
        t_px_cm    = t_px + gamma2*bp*bx - gamma*bx*t_e
        t_py_cm    = t_py + gamma2*bp*by - gamma*by*t_e
        t_pz_cm    = t_pz + gamma2*bp*bz - gamma*bz*t_e
        t_p_cm     = np.sqrt(t_px_cm**2 + t_py_cm**2 + t_pz_cm**2)
        cos_theta_star = np.where(t_p_cm > 0, t_pz_cm / t_p_cm, 0.)

        _feats = {
            "lep_pt"        : leps.pt.to_numpy(),
            "lep_eta"       : leps.eta.to_numpy(),
            "lep_phi"       : leps.phi.to_numpy(),
            "lep_mass"      : leps.mass.to_numpy(),
            "met_pt"        : met.pt.to_numpy(),
            "met_phi"       : met.phi.to_numpy(),
            "jet0_pt"       :   jets.pt[:,0].to_numpy(),
            "jet0_eta"      :  jets.eta[:,0].to_numpy(),
            "jet0_phi"      :  jets.phi[:,0].to_numpy(),
            "jet0_mass"     : jets.mass[:,0].to_numpy(),
            "jet1_pt"       :   jets.pt[:,1].to_numpy(),
            "jet1_eta"      :  jets.eta[:,1].to_numpy(),
            "jet1_phi"      :  jets.phi[:,1].to_numpy(),
            "jet1_mass"     : jets.mass[:,1].to_numpy(),
            "jet2_pt"       :   jets.pt[:,2].to_numpy(),
            "jet2_eta"      :  jets.eta[:,2].to_numpy(),
            "jet2_phi"      :  jets.phi[:,2].to_numpy(),
            "jet2_mass"     : jets.mass[:,2].to_numpy(),
            "jet3_pt"       :   jets.pt[:,3].to_numpy(),
            "jet3_eta"      :  jets.eta[:,3].to_numpy(),
            "jet3_phi"      :  jets.phi[:,3].to_numpy(),
            "jet3_mass"     : jets.mass[:,3].to_numpy(),
            "njets"         : ak.num(jets).to_numpy(),
            "HT"            : ak.sum(jets.pt,axis=1).to_numpy(),
            "mT_W"          : mT_W,
            "jet0_flav"     : jets.hadronFlavour[:,0].to_numpy().astype(float),
            "jet1_flav"     : jets.hadronFlavour[:,1].to_numpy().astype(float),
            "jet2_flav"     : jets.hadronFlavour[:,2].to_numpy().astype(float),
            "jet3_flav"     : jets.hadronFlavour[:,3].to_numpy().astype(float),
            "lep_top_pt"    : np.where(lep_is_neg, tb_pt,   t_pt),    # lep_top pt
            "lep_top_eta"   : np.where(lep_is_neg, tb_eta,  t_eta),   # lep_top eta
            "lep_top_phi"   : np.where(lep_is_neg, tb_phi,  t_phi),   # lep_top phi
            "lep_top_mass"  : np.where(lep_is_neg, tb_mass, t_mass),  # lep_top mass
            "had_top_pt"    : np.where(lep_is_neg, t_pt,   tb_pt),    # had_top pt
            "had_top_eta"   : np.where(lep_is_neg, t_eta,  tb_eta),   # had_top eta
            "had_top_phi"   : np.where(lep_is_neg, t_phi,  tb_phi),   # had_top phi
            "had_top_mass"  : np.where(lep_is_neg, t_mass, tb_mass),  # had_top mass
            "dr_tt"         : dr_tt,           # ΔR(t, t̄)
            "dr_lep_had"    : dr_lep_had,      # ΔR(l, had_top)
            "m_ttbar"       : m_ttbar,         # M(tt̄)
            "cos_theta_star": cos_theta_star,  # cos θ* (production angle in ttbar CM frame)
        }

        _weights  = _direct_weights(events, mask) if with_weights else {}

        per_file.append(_feats)
        per_file_w.append(_weights)

    features = {
        name: np.concatenate([d[name] for d in per_file]).astype(np.float64)
        for name in FEATURE_NAMES
    }

    weights = {}
    if with_weights:
        weights = {
            name: np.concatenate([d[name] for d in per_file_w]).astype(np.float64)
            for name in WC_NAMES
        }

    sample: Sample = {"features":features, "weights":weights}
    validate_sample(sample, require_weights=with_weights)

    return sample


def load_weights(files, *, nevents=None):
    """weight 軸專用:在同一份 nano 上產出逐 event 對齊的 direct & poly weight。
    回傳 {"direct": {...17...}, "poly": {...17...}},兩者 per-event 對齊。
    不走 Sample 契約(weight 軸是 source 內比較,跟 feature 的跨 source 不同)。
    """
    direct_per_file, poly_per_file = [], []
    for f in files:
        events = NanoEventsFactory.from_root({f: "Events"}, schemaclass=NanoAODSchema).events()
        if nevents:
            events = events[:nevents]
        cleanleps, cleanjets = genObjectSelection(events)
        mask = genEventSelection(cleanleps, cleanjets)

        direct_per_file.append(_direct_weights(events, mask))
        poly_per_file.append(_poly_weights(events, mask))

    direct = {n: np.concatenate([d[n] for d in direct_per_file]) for n in WC_NAMES}
    poly   = {n: np.concatenate([d[n] for d in poly_per_file])   for n in WC_NAMES}
    return {"direct": direct, "poly": poly}


def _direct_weights(events, mask) -> dict:
    """對 SM + 16 operator 取 direct LHEWeight,回傳 weights dict。

    - SM    : events.LHEWeight["sm_point"][mask]
    - op_i  : events.LHEWeight[rwgt_field_i][mask],rwgt_field 由 find_linear_rwgts 解析。
    """

    def find_linear_rwgts(events):
        """Map operator -> LHEWeight field name for the 16 single-operator-on points
        (EFTrwgt201..EFTrwgt216), by parsing each field's encoded coefficient list."""
        pattern = re.compile(r"^EFTrwgt(\d+)_(.*)$")
        rwgts = {}
        for f in events.LHEWeight.fields:
            m = pattern.match(f)
            if not m:
                continue
            n = int(m.group(1))
            if not (201 <= n <= 216):
                continue
            tokens = m.group(2).split("_")
            # tokens alternate (operator_name, value_string) like "ctGRe", "1p", "ctGIm", "0p", ...
            for i in range(0, len(tokens) - 1, 2):
                op, val = tokens[i], tokens[i + 1]
                if val == "1p":
                    rwgts[op] = f
                    break
        missing = [op for op in OPERATORS if op not in rwgts]
        if missing:
            raise RuntimeError(f"Could not find linear-on rwgt for: {missing}")

        # for key, value in rwgts.items():
        #     print(f"Linear weights: {key}, {value}")

        return rwgts

    rwgts = find_linear_rwgts(events)

    # Weights for SM + 16 BSM
    w_sm = ak.to_numpy(events.LHEWeight[SM_NAME][mask]).astype(np.float64)
    w_bsm_dict = {
        op: ak.to_numpy(events.LHEWeight[rwgts[op]][mask]).astype(np.float64)
        for op in OPERATORS
    }

    weights = {SM_NAME: w_sm}      # 先放 SM
    weights.update(w_bsm_dict)     # 再併入 16 個 operator

    return weights

def _poly_weights(events, mask) -> dict:
    """
    對 SM + 16 operator,在 nano 上用 calculate_fit 解係數再多項式重建 poly weight。
    回傳 {WC_NAMES: (n_sel,) poly weight}。與 _direct_weights 逐 event 對齊,供 compare_weights 比對。
    """

    from pretraining.fitCoefficients import calculate_fit, vec, wcs as FIT_WCS

    fields = events.LHEWeight.fields
    eft_coeffs, _ = calculate_fit(fields, events.LHEWeight[mask])   # (n_pairs, n_sel)
    coefs = eft_coeffs.T                                            # (n_sel, n_pairs)

    # # Build WC vectors using the CORRECT vec convention (with sqrt(2))
    # c0 = np.zeros(n_wc); c0[FIT_WCS.index('cSM')] = 1.0
    # c1 = np.zeros(n_wc); c1[FIT_WCS.index('cSM')] = 1.0
    # c1[FIT_WCS.index('ctGRe')] = 1.0
    #
    # w_sm_poly  = coefs @ vec(np.outer(c0, c0))
    # w_eft_poly = coefs @ vec(np.outer(c1, c1))

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
