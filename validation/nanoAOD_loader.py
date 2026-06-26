"""
nanoAOD_loader.py — 從 NanoAOD root 檔獨立重算 features / weights,回傳 Sample。

職責:
  - selection:import 共享的 analysis_tools(genObjectSelection / genEventSelection),
               **不可**抄 eft_sensitivity_scan.py 裡 local 那份 — selection 不一致是 noise,
               必須與 ml_data 用同一份。
  - features :獨立重算 72 個 feature(這是受測對象,必須與 tensor 側互不依賴)。
               index 0-40 沿用既有邏輯;41-71 為新增(Tier 2 / Tier 1 / 3D 基底 / Tier 3)。
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


# ─────────────────────────────────────────────────────────────────────────────
# Kinematic helpers — independent recompute of the new (index 41-71) features.
# Same physics as pretraining.ml_data, written here as a parallel implementation
# (the validation framework keeps the two sides separate on purpose).
# ─────────────────────────────────────────────────────────────────────────────
def _boost(px, py, pz, e, bx, by, bz):
    """Boost a 4-vector into a frame moving with velocity (bx,by,bz)."""
    b2 = np.clip(bx * bx + by * by + bz * bz, 0., 1. - 1e-9)
    g = 1. / np.sqrt(1. - b2)
    bp = bx * px + by * py + bz * pz
    g2 = np.where(b2 > 1e-12, (g - 1.) / np.where(b2 > 1e-12, b2, 1.), 0.)
    return (px + g2 * bp * bx - g * e * bx,
            py + g2 * bp * by - g * e * by,
            pz + g2 * bp * bz - g * e * bz,
            g * (e - bp))


def _rap(e, pz):
    return 0.5 * np.log(np.clip(e + pz, 1e-10, None) / np.clip(e - pz, 1e-10, None))


def _cart(pt, eta, phi, mass):
    px = pt * np.cos(phi); py = pt * np.sin(phi); pz = pt * np.sinh(eta)
    return px, py, pz, np.sqrt(px * px + py * py + pz * pz + mass * mass)


def _rest_dir(apx, apy, apz, ae, tpx, tpy, tpz, te):
    inv = 1. / np.maximum(te, 1e-10)
    rx, ry, rz, _ = _boost(apx, apy, apz, ae, tpx * inv, tpy * inv, tpz * inv)
    mag = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), 1e-10)
    return rx / mag, ry / mag, rz / mag


def _cos_hel(apx, apy, apz, ae, tpx, tpy, tpz, te, sbx, sby, sbz):
    ux, uy, uz = _rest_dir(apx, apy, apz, ae, tpx, tpy, tpz, te)
    hx, hy, hz, _ = _boost(tpx, tpy, tpz, te, sbx, sby, sbz)
    hmag = np.maximum(np.sqrt(hx * hx + hy * hy + hz * hz), 1e-10)
    return np.clip((ux * hx + uy * hy + uz * hz) / hmag, -1., 1.)


def _dr(eta1, phi1, eta2, phi2):
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def _pmass(v1, v2):
    px = v1[0] + v2[0]; py = v1[1] + v2[1]; pz = v1[2] + v2[2]; e = v1[3] + v2[3]
    return np.sqrt(np.maximum(e * e - px * px - py * py - pz * pz, 0.))


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
        3. 重算 72 個 feature 成 features dict(np.ndarray)。
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

        # ── new features (index 41-54): boosts, spin angles, {n,r,k} basis ──
        lep_pt_np   = ak.to_numpy(leps.pt).astype(float)
        lep_mass_np = ak.to_numpy(leps.mass).astype(float)
        lep_px, lep_py, lep_pz, lep_e = _cart(lep_pt_np, lep_eta_np, lep_phi_np, lep_mass_np)

        # leptonic / hadronic top 4-vectors (lep_is_neg: l- ⇒ lep top = tbar)
        leptop_px = np.where(lep_is_neg, tb_px, t_px);  leptop_py = np.where(lep_is_neg, tb_py, t_py)
        leptop_pz = np.where(lep_is_neg, tb_pz, t_pz);  leptop_e  = np.where(lep_is_neg, tb_e,  t_e)
        hadtop_px = np.where(lep_is_neg, t_px, tb_px);  hadtop_py = np.where(lep_is_neg, t_py, tb_py)
        hadtop_pz = np.where(lep_is_neg, t_pz, tb_pz);  hadtop_e  = np.where(lep_is_neg, t_e,  tb_e)

        # hadronic spin analyzer: the unique hard-process down-type quark (d/s)
        down = ak.pad_none(gp[is_final & ((abs(gp.pdgId) == 1) | (abs(gp.pdgId) == 3))], 1)[:, 0]
        down_px, down_py, down_pz, down_e = _cart(
            to_np(down.pt), to_np(down.eta), to_np(down.phi), to_np(down.mass))

        inv_sys_e = 1. / np.maximum(sys_e, 1e-10)
        sbx, sby, sbz = sys_px * inv_sys_e, sys_py * inv_sys_e, sys_pz * inv_sys_e

        cos_theta_l   = _cos_hel(lep_px, lep_py, lep_pz, lep_e,
                                 leptop_px, leptop_py, leptop_pz, leptop_e, sbx, sby, sbz)
        cos_theta_had = _cos_hel(down_px, down_py, down_pz, down_e,
                                 hadtop_px, hadtop_py, hadtop_pz, hadtop_e, sbx, sby, sbz)
        c_hel = cos_theta_l * cos_theta_had

        dy_tt = _rap(t_e, t_pz) - _rap(tb_e, tb_pz)
        pt_tt = np.sqrt(sys_px**2 + sys_py**2)
        y_tt  = _rap(sys_e, sys_pz)

        # {n,r,k} basis from top in tt̄ CM frame (t_*_cm already computed above)
        k_mag = np.maximum(np.sqrt(t_px_cm**2 + t_py_cm**2 + t_pz_cm**2), 1e-10)
        kx, ky, kz = t_px_cm / k_mag, t_py_cm / k_mag, t_pz_cm / k_mag
        sinT = np.sqrt(np.maximum(1. - kz * kz, 0.))
        inv_sinT = np.where(sinT > 1e-6, 1. / np.maximum(sinT, 1e-10), 0.)
        nx, ny, nz = -ky * inv_sinT, kx * inv_sinT, np.zeros_like(kx)
        rx, ry, rz = -kz * kx * inv_sinT, -kz * ky * inv_sinT, (1. - kz * kz) * inv_sinT
        lux, luy, luz = _rest_dir(lep_px, lep_py, lep_pz, lep_e,
                                  leptop_px, leptop_py, leptop_pz, leptop_e)
        hux, huy, huz = _rest_dir(down_px, down_py, down_pz, down_e,
                                  hadtop_px, hadtop_py, hadtop_pz, hadtop_e)
        cos_lep_n = np.clip(lux * nx + luy * ny + luz * nz, -1., 1.)
        cos_lep_r = np.clip(lux * rx + luy * ry + luz * rz, -1., 1.)
        cos_lep_k = np.clip(lux * kx + luy * ky + luz * kz, -1., 1.)
        cos_had_n = np.clip(hux * nx + huy * ny + huz * nz, -1., 1.)
        cos_had_r = np.clip(hux * rx + huy * ry + huz * rz, -1., 1.)
        cos_had_k = np.clip(hux * kx + huy * ky + huz * kz, -1., 1.)

        # ── Tier 3 (index 55-71): pairwise lepton-jet / jet-jet geometry & masses ──
        j_eta = [ak.to_numpy(jets.eta[:, i]).astype(float)  for i in range(4)]
        j_phi = [ak.to_numpy(jets.phi[:, i]).astype(float)  for i in range(4)]
        j_pt  = [ak.to_numpy(jets.pt[:, i]).astype(float)   for i in range(4)]
        j_m   = [ak.to_numpy(jets.mass[:, i]).astype(float) for i in range(4)]
        j_fl  = [ak.to_numpy(jets.hadronFlavour[:, i])      for i in range(4)]
        j_v   = [_cart(j_pt[i], j_eta[i], j_phi[i], j_m[i]) for i in range(4)]
        lep_v = (lep_px, lep_py, lep_pz, lep_e)

        dr_l_j = [_dr(lep_eta_np, lep_phi_np, j_eta[i], j_phi[i]) for i in range(4)]
        _pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        dr_jj = [_dr(j_eta[i], j_phi[i], j_eta[j], j_phi[j]) for (i, j) in _pairs]
        m_jj  = [_pmass(j_v[i], j_v[j]) for (i, j) in _pairs]
        _m_lb = np.stack([_pmass(lep_v, j_v[i]) for i in range(4)])
        _is_b = np.stack([j_fl[i] == 5 for i in range(4)])
        m_lb_min = np.min(np.where(_is_b, _m_lb, np.inf), axis=0)
        m_lb_min = np.where(np.isfinite(m_lb_min), m_lb_min, 0.)

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
            # ── Tier 2: tt̄ system kinematics ──
            "dy_tt"         : dy_tt,
            "dphi_tt"       : dphi_tt,
            "pt_tt"         : pt_tt,
            "y_tt"          : y_tt,
            # ── Tier 1: spin correlation / polarization ──
            "cos_theta_l"   : cos_theta_l,
            "cos_theta_had" : cos_theta_had,
            "c_hel"         : c_hel,
            "dphi_l_had"    : dphi_lh,         # Δφ(l, had_top), lab
            # ── 3D spin-basis projections (common {n,r,k}) ──
            "cos_lep_n"     : cos_lep_n,
            "cos_lep_r"     : cos_lep_r,
            "cos_lep_k"     : cos_lep_k,
            "cos_had_n"     : cos_had_n,
            "cos_had_r"     : cos_had_r,
            "cos_had_k"     : cos_had_k,
            # ── Tier 3: pairwise geometry & masses ──
            "dr_l_j0"       : dr_l_j[0],
            "dr_l_j1"       : dr_l_j[1],
            "dr_l_j2"       : dr_l_j[2],
            "dr_l_j3"       : dr_l_j[3],
            "dr_j01"        : dr_jj[0],
            "dr_j02"        : dr_jj[1],
            "dr_j03"        : dr_jj[2],
            "dr_j12"        : dr_jj[3],
            "dr_j13"        : dr_jj[4],
            "dr_j23"        : dr_jj[5],
            "m_j01"         : m_jj[0],
            "m_j02"         : m_jj[1],
            "m_j03"         : m_jj[2],
            "m_j12"         : m_jj[3],
            "m_j13"         : m_jj[4],
            "m_j23"         : m_jj[5],
            "m_lb_min"      : m_lb_min,
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
