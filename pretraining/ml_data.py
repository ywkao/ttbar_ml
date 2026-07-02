import awkward as ak
import numpy as np
from torch import float64, tensor, from_numpy
from coffea.nanoevents import NanoAODSchema
from coffea.processor import ProcessorABC, defaultdict_accumulator

from .fitCoefficients import calculate_fit
from .analysis_tools import genObjectSelection, genEventSelection, topObjectSelection
from .TensorAccumulator import TensorAccumulator

NanoAODSchema.warn_missing_crossrefs = False


def _boost_into(px, py, pz, e, bx, by, bz):
    """Boost a 4-vector (px,py,pz,e) into a frame moving with velocity (bx,by,bz)
    relative to the current frame. All inputs are numpy arrays of equal shape.
    Returns (px', py', pz', e')."""
    b2 = bx * bx + by * by + bz * bz
    b2 = np.clip(b2, 0., 1. - 1e-9)
    gamma = 1. / np.sqrt(1. - b2)
    bp = bx * px + by * py + bz * pz
    safe_b2 = np.where(b2 > 1e-12, b2, 1.)
    gamma2 = np.where(b2 > 1e-12, (gamma - 1.) / safe_b2, 0.)
    px2 = px + gamma2 * bp * bx - gamma * e * bx
    py2 = py + gamma2 * bp * by - gamma * e * by
    pz2 = pz + gamma2 * bp * bz - gamma * e * bz
    e2 = gamma * (e - bp)
    return px2, py2, pz2, e2


def _rapidity(e, pz):
    """Rapidity y = 0.5 ln[(E+pz)/(E-pz)], guarded against div-by-zero / log(0)."""
    num = np.clip(e + pz, 1e-10, None)
    den = np.clip(e - pz, 1e-10, None)
    return 0.5 * np.log(num / den)


def _cos_helicity(a_px, a_py, a_pz, a_e,
                  top_px, top_py, top_pz, top_e,
                  sys_bx, sys_by, sys_bz):
    """cos of the angle, in the parent-top rest frame, between a spin-analyzer's
    direction and the helicity axis (the parent-top flight direction in the tt̄ CM
    frame). The analyzer is boosted lab -> tt̄ ZMF -> top rest frame; the top is
    boosted into the tt̄ CM frame to define the axis. Returns cosθ in [-1, 1]."""
    ux, uy, uz = _rest_frame_dir(a_px, a_py, a_pz, a_e,
                                 top_px, top_py, top_pz, top_e,
                                 sys_bx, sys_by, sys_bz)
    hx, hy, hz, _ = _boost_into(top_px, top_py, top_pz, top_e, sys_bx, sys_by, sys_bz)
    hmag = np.maximum(np.sqrt(hx * hx + hy * hy + hz * hz), 1e-10)
    cos = (ux * hx + uy * hy + uz * hz) / hmag
    return np.clip(cos, -1., 1.)


def _rest_frame_dir(a_px, a_py, a_pz, a_e, top_px, top_py, top_pz, top_e,
                    sys_bx, sys_by, sys_bz):
    """Unit direction of an analyzer in its parent-top rest frame, reached via
    the tt̄ ZMF (lab -> ZMF -> top rest). The two-step chain is required: a
    direct lab -> top boost differs from it by a Wigner rotation.
    Returns (ux, uy, uz) numpy arrays."""
    ax, ay, az, ae = _boost_into(a_px, a_py, a_pz, a_e, sys_bx, sys_by, sys_bz)
    tx, ty, tz, te = _boost_into(top_px, top_py, top_pz, top_e, sys_bx, sys_by, sys_bz)
    inv_te = 1. / np.maximum(te, 1e-10)
    rx, ry, rz, _ = _boost_into(ax, ay, az, ae,
                                tx * inv_te, ty * inv_te, tz * inv_te)
    mag = np.maximum(np.sqrt(rx * rx + ry * ry + rz * rz), 1e-10)
    return rx / mag, ry / mag, rz / mag


def _to_cartesian(pt, eta, phi, mass):
    """(pt, eta, phi, mass) numpy arrays -> Cartesian (px, py, pz, E)."""
    eta = np.clip(eta, -10., 10.)
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e  = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return px, py, pz, e


def _beta_t_star(tl_px, tl_py, tl_pz, tl_e, th_px, th_py, th_pz, th_e):
    """β* of the leptonic top in the hadronic-top rest frame (relative velocity
    of the two tops); boost t_ℓ into t_h's rest frame and take |p|/E. ∈ [0, 1)."""
    inv_e = 1. / np.maximum(th_e, 1e-10)
    px2, py2, pz2, e2 = _boost_into(tl_px, tl_py, tl_pz, tl_e,
                                    th_px * inv_e, th_py * inv_e, th_pz * inv_e)
    p_mag = np.sqrt(px2**2 + py2**2 + pz2**2)
    return np.clip(p_mag / np.maximum(e2, 1e-10), 0., 1.)


def _delta_r(eta1, phi1, eta2, phi2):
    """ΔR = sqrt(Δη² + Δφ²) with Δφ wrapped to (-π, π]."""
    deta = eta1 - eta2
    dphi = np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))
    return np.sqrt(deta * deta + dphi * dphi)


def _pair_mass(v1, v2):
    """Invariant mass of two 4-vectors given as (px, py, pz, e) tuples."""
    px = v1[0] + v2[0];  py = v1[1] + v2[1]
    pz = v1[2] + v2[2];  e  = v1[3] + v2[3]
    return np.sqrt(np.maximum(e * e - px * px - py * py - pz * pz, 0.))


class SemiLepProcessor(ProcessorABC):
    def __init__(self, runFit=True, dtype=float64):
        self.runFit = runFit
        self._dtype = float64
        #self.coefficients = coefficients
       
    def accumulator(self):
        return self._accumulator
        
    def process(self, events):   
        metadata = defaultdict_accumulator(float)

        # Selection
        cleanleps, cleanjets = genObjectSelection(events)
        event_mask = genEventSelection(cleanleps, cleanjets)
        leps = ak.flatten(cleanleps[event_mask])
        jets = cleanjets[event_mask]
        met = events.GenMET[event_mask]

        # Determining the structure coefficients
        fit_coefs = TensorAccumulator(tensor([]), dtype=self._dtype)
        the_weights = TensorAccumulator(tensor([]), dtype=self._dtype)
        residuals = TensorAccumulator(tensor([]), dtype=self._dtype)
        if self.runFit:
            # Using the weights to fit for coefficients
            eft_coeffs, _ = calculate_fit(events.LHEWeight.fields, events.LHEWeight[event_mask])
        elif hasattr(events, "EFTfitCoefficients"):
            # Using the precalculated coefficients
            eft_coeffs = ak.to_numpy(events.EFTfitCoefficients)
            eft_coeffs = eft_coeffs[event_mask] if eft_coeffs is not None else None
        else: # Otherwise neglect
            eft_coeffs = None

        # Creating tensors
        features = self.calc_features(leps, met, jets, events.GenPart[event_mask])
        fit_coefs = fit_coefs.concat(from_numpy(eft_coeffs.T))

        # Storing meta info
        metadata["InputEventCount"] = len(events)
        metadata["SelectedEventCount"] = ak.sum(event_mask)
        metadata["sumGenWeights"] = ak.sum(events.genWeight)
        metadata["sumSMreweights"] = ak.sum(events.LHEWeight["sm_point"])

        return {
            'features': features, 
            'fit_coefs': fit_coefs, 
            'residuals': residuals,
            'weights': the_weights,
            'metadata': metadata
        }
    
    def calc_features(self, leps, met, jets, genparts):
        features  = TensorAccumulator(tensor([]), dtype=self._dtype)

        #lep_jets = leps+jets[:,0]+jets[:,1]+jets[:,2]+jets[:,3]
        #HT = ak.sum(jets.pt,axis=1)

        #casted = ak.broadcast_arrays(leps,jets, depth_limit=2)
        #sum_lj0 = casted[0]+casted[1]
        #max_lj0 = ak.argmax(sum_lj0.pt, axis=1, keepdims=True)

        lep_phi = leps.phi.to_numpy().astype(float)
        met_pt  = met.pt.to_numpy().astype(float)
        met_phi = met.phi.to_numpy().astype(float)
        mT_W = np.sqrt(2 * leps.pt.to_numpy().astype(float) * met_pt
                       * (1 - np.cos(lep_phi - met_phi)))

        features = features.concat(from_numpy(np.concatenate([[leps.pt.to_numpy()],
                                                            [leps.eta.to_numpy()],
                                                            [leps.phi.to_numpy()],
                                                            [leps.mass.to_numpy()],
                                                            [met.pt.to_numpy()],
                                                            [met.phi.to_numpy()],
                                                            [  jets.pt[:,0].to_numpy()],
                                                            [ jets.eta[:,0].to_numpy()],
                                                            [ jets.phi[:,0].to_numpy()],
                                                            [jets.mass[:,0].to_numpy()],
                                                            [  jets.pt[:,1].to_numpy()],
                                                            [ jets.eta[:,1].to_numpy()],
                                                            [ jets.phi[:,1].to_numpy()],
                                                            [jets.mass[:,1].to_numpy()],
                                                            [  jets.pt[:,2].to_numpy()],
                                                            [ jets.eta[:,2].to_numpy()],
                                                            [ jets.phi[:,2].to_numpy()],
                                                            [jets.mass[:,2].to_numpy()],
                                                            [  jets.pt[:,3].to_numpy()],
                                                            [ jets.eta[:,3].to_numpy()],
                                                            [ jets.phi[:,3].to_numpy()],
                                                            [jets.mass[:,3].to_numpy()],
                                                            [ak.num(jets).to_numpy()],
                                                            [ak.sum(jets.pt,axis=1).to_numpy()],
                                                            [mT_W],
                                                            [jets.hadronFlavour[:,0].to_numpy().astype(float)],
                                                            [jets.hadronFlavour[:,1].to_numpy().astype(float)],
                                                            [jets.hadronFlavour[:,2].to_numpy().astype(float)],
                                                            [jets.hadronFlavour[:,3].to_numpy().astype(float)],
                                                            *self.calc_top_features(genparts, leps),
                                                            *self.calc_pair_features(leps, jets),
                                                            #[lep_jets.mass.to_numpy()],
                                                            #[leps.delta_phi(jets[:,0]).to_numpy()],
                                                            #[leps.delta_phi(jets[:,1]).to_numpy()],
                                                            #[(jets[:,0].rho/jets[:,1].rho).to_numpy()],
                                                            #[ak.flatten(sum_lj0.pt[max_lj0]).to_numpy()],#TOP-22-006
                                                            #[ak.flatten(leps.delta_r(jets[max_lj0])).to_numpy()], #TOP-22-006
                                                            ]).T))
        return features
    
    def calc_top_features(self, genparts, leps):
        """Returns a list of 1D numpy arrays, one per feature, for use in np.concatenate."""
        is_final = genparts.hasFlags(["fromHardProcess", "isLastCopy"])
        t    = ak.pad_none(genparts[is_final & (genparts.pdgId ==  6)], 1)[:,0]
        tbar = ak.pad_none(genparts[is_final & (genparts.pdgId == -6)], 1)[:,0]

        # l- (pdgId=11,13 > 0) comes from W- from tbar; l+ (pdgId<0) comes from W+ from t
        lep_is_negative = leps.pdgId > 0

        def sel(cond, a, b):
            return [ak.to_numpy(ak.fill_none(ak.where(cond, a, b), 0.)).astype(float)]

        def to_np(arr):
            return ak.to_numpy(ak.fill_none(arr, 0.)).astype(float)

        # Extract top 4-vector components as numpy arrays
        t_pt   = to_np(t.pt);   t_eta  = to_np(t.eta)
        t_phi  = to_np(t.phi);  t_mass = to_np(t.mass)
        tb_pt  = to_np(tbar.pt);  tb_eta = to_np(tbar.eta)
        tb_phi = to_np(tbar.phi); tb_mass = to_np(tbar.mass)

        # Convert (pt, eta, phi, mass) -> Cartesian 4-vectors
        t_px  = t_pt  * np.cos(t_phi);    t_py  = t_pt  * np.sin(t_phi)
        t_pz  = t_pt  * np.sinh(t_eta);   t_e   = np.sqrt(t_px**2 + t_py**2 + t_pz**2 + t_mass**2)
        tb_px = tb_pt * np.cos(tb_phi);   tb_py = tb_pt * np.sin(tb_phi)
        tb_pz = tb_pt * np.sinh(tb_eta);  tb_e  = np.sqrt(tb_px**2 + tb_py**2 + tb_pz**2 + tb_mass**2)

        # ttbar system 4-vector
        sys_px = t_px + tb_px;  sys_py = t_py + tb_py
        sys_pz = t_pz + tb_pz;  sys_e  = t_e  + tb_e

        # M(tt̄) — invariant mass of the ttbar system
        m_ttbar = np.sqrt(np.maximum(sys_e**2 - sys_px**2 - sys_py**2 - sys_pz**2, 0.))

        # ΔR(t, t̄)
        deta_tt = t_eta - tb_eta
        dphi_tt = np.arctan2(np.sin(t_phi - tb_phi), np.cos(t_phi - tb_phi))
        dr_tt   = np.sqrt(deta_tt**2 + dphi_tt**2)

        # ΔR(l, had_top)  —  had_top is t when lep_is_negative, else tbar
        had_top_eta = to_np(ak.where(lep_is_negative, t.eta,  tbar.eta))
        had_top_phi = to_np(ak.where(lep_is_negative, t.phi,  tbar.phi))
        lep_eta_np  = leps.eta.to_numpy().astype(float)
        lep_phi_np  = leps.phi.to_numpy().astype(float)
        deta_lh    = lep_eta_np - had_top_eta
        dphi_lh    = np.arctan2(np.sin(lep_phi_np - had_top_phi), np.cos(lep_phi_np - had_top_phi))
        dr_lep_had = np.sqrt(deta_lh**2 + dphi_lh**2)

        # cos θ* — angle of t in the ttbar CM frame w.r.t. beam axis (z)
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

        # ── lepton 4-vector (lab, Cartesian); lep_eta/phi already derived above ──
        lep_pt   = leps.pt.to_numpy().astype(float)
        lep_mass = leps.mass.to_numpy().astype(float)
        lep_px = lep_pt * np.cos(lep_phi_np);  lep_py = lep_pt * np.sin(lep_phi_np)
        lep_pz = lep_pt * np.sinh(lep_eta_np)
        lep_e  = np.sqrt(lep_px**2 + lep_py**2 + lep_pz**2 + lep_mass**2)

        # ── leptonic / hadronic top 4-vectors (lab, Cartesian) ──
        # l- (lep_is_negative) comes from tbar's W-, so the leptonic top is tbar
        # and the hadronic top is t; for l+ the roles swap.
        lep_from_tbar = ak.to_numpy(lep_is_negative).astype(bool)
        leptop_px = np.where(lep_from_tbar, tb_px, t_px)
        leptop_py = np.where(lep_from_tbar, tb_py, t_py)
        leptop_pz = np.where(lep_from_tbar, tb_pz, t_pz)
        leptop_e  = np.where(lep_from_tbar, tb_e,  t_e)
        hadtop_px = np.where(lep_from_tbar, t_px, tb_px)
        hadtop_py = np.where(lep_from_tbar, t_py, tb_py)
        hadtop_pz = np.where(lep_from_tbar, t_pz, tb_pz)
        hadtop_e  = np.where(lep_from_tbar, t_e,  tb_e)

        # ── hadronic spin analyzer: down-type quark (d/s) from the hadronic W ──
        # Require the quark's parent to be a W: the bare |pdgId|∈{1,3} selection also
        # tags INITIAL-STATE (status-21) d/s partons from qq̄ initiation, which sit at
        # the front of GenPart and lie along the beam (pt≈0, |eta|→∞ → sinh overflow).
        # The W-parent cut keeps exactly the hadronic-W down-type quark.
        down_all   = genparts[is_final & ((abs(genparts.pdgId) == 1) |
                                          (abs(genparts.pdgId) == 3))]
        down_from_w = ak.fill_none(abs(down_all.distinctParent.pdgId) == 24, False)
        down_quark = ak.pad_none(down_all[down_from_w], 1)[:, 0]
        down_pt   = to_np(down_quark.pt);   down_eta  = np.clip(to_np(down_quark.eta), -10., 10.)
        down_phi  = to_np(down_quark.phi);  down_mass = to_np(down_quark.mass)
        down_px = down_pt * np.cos(down_phi);  down_py = down_pt * np.sin(down_phi)
        down_pz = down_pt * np.sinh(down_eta)
        down_e  = np.sqrt(down_px**2 + down_py**2 + down_pz**2 + down_mass**2)

        # tt̄ CM boost vector (lab → tt̄ rest frame)
        inv_sys_e = 1. / np.maximum(sys_e, 1e-10)
        sys_bx = sys_px * inv_sys_e;  sys_by = sys_py * inv_sys_e;  sys_bz = sys_pz * inv_sys_e

        # ── Tier 1: spin correlation / polarization (helicity frame) ──
        cos_theta_l   = _cos_helicity(lep_px, lep_py, lep_pz, lep_e,
                                      leptop_px, leptop_py, leptop_pz, leptop_e,
                                      sys_bx, sys_by, sys_bz)
        cos_theta_had = _cos_helicity(down_px, down_py, down_pz, down_e,
                                      hadtop_px, hadtop_py, hadtop_pz, hadtop_e,
                                      sys_bx, sys_by, sys_bz)
        dphi_l_had = dphi_lh   # Δφ(l, had_top), lab (reuse signed value from above)

        # ── Tier 2: tt̄ system kinematics ──
        dy_tt = _rapidity(t_e, t_pz) - _rapidity(tb_e, tb_pz)   # Δy(t, t̄)
        pt_tt = np.sqrt(sys_px**2 + sys_py**2)                  # pT(tt̄)
        y_tt  = _rapidity(sys_e, sys_pz)                        # y(tt̄)

        # ── {n, r, k} spin basis (Bernreuther/ATLAS convention) ──
        # k̂ = top (t) direction in the tt̄ CM frame (t_*_cm computed above);
        # p̂ = beam = +ẑ;  n̂ = (p̂×k̂)/sinΘ;  r̂ = (p̂ − cosΘ k̂)/sinΘ.
        k_mag = np.maximum(np.sqrt(t_px_cm**2 + t_py_cm**2 + t_pz_cm**2), 1e-10)
        kx, ky, kz = t_px_cm / k_mag, t_py_cm / k_mag, t_pz_cm / k_mag   # cosΘ = kz
        sinT = np.sqrt(np.maximum(1. - kz * kz, 0.))
        inv_sinT = np.where(sinT > 1e-6, 1. / np.maximum(sinT, 1e-10), 0.)  # 0 when t∥beam
        nx, ny, nz = -ky * inv_sinT, kx * inv_sinT, np.zeros_like(kx)
        rx, ry, rz = -kz * kx * inv_sinT, -kz * ky * inv_sinT, (1. - kz * kz) * inv_sinT

        # analyzer directions in their parent-top rest frames
        lep_ux, lep_uy, lep_uz = _rest_frame_dir(lep_px, lep_py, lep_pz, lep_e,
                                                 leptop_px, leptop_py, leptop_pz, leptop_e,
                                                 sys_bx, sys_by, sys_bz)
        had_ux, had_uy, had_uz = _rest_frame_dir(down_px, down_py, down_pz, down_e,
                                                 hadtop_px, hadtop_py, hadtop_pz, hadtop_e,
                                                 sys_bx, sys_by, sys_bz)

        # project each analyzer onto the common {n, r, k} basis
        cos_lep_n = np.clip(lep_ux * nx + lep_uy * ny + lep_uz * nz, -1., 1.)
        cos_lep_r = np.clip(lep_ux * rx + lep_uy * ry + lep_uz * rz, -1., 1.)
        cos_lep_k = np.clip(lep_ux * kx + lep_uy * ky + lep_uz * kz, -1., 1.)
        cos_had_n = np.clip(had_ux * nx + had_uy * ny + had_uz * nz, -1., 1.)
        cos_had_r = np.clip(had_ux * rx + had_uy * ry + had_uz * rz, -1., 1.)
        cos_had_k = np.clip(had_ux * kx + had_uy * ky + had_uz * kz, -1., 1.)

        # ── opening-angle spin observables (rest-frame analyzer directions) ──
        c_hel = np.clip(lep_ux * had_ux + lep_uy * had_uy + lep_uz * had_uz, -1., 1.)
        c_han = np.clip(c_hel - 2. * cos_lep_k * cos_had_k, -1., 1.)
        beta_t_star = _beta_t_star(leptop_px, leptop_py, leptop_pz, leptop_e,
                                   hadtop_px, hadtop_py, hadtop_pz, hadtop_e)

        return [
            sel(lep_is_negative, tbar.pt,   t.pt),    # lep_top pt
            sel(lep_is_negative, tbar.eta,  t.eta),   # lep_top eta
            sel(lep_is_negative, tbar.phi,  t.phi),   # lep_top phi
            sel(lep_is_negative, tbar.mass, t.mass),  # lep_top mass
            sel(lep_is_negative, t.pt,   tbar.pt),    # had_top pt
            sel(lep_is_negative, t.eta,  tbar.eta),   # had_top eta
            sel(lep_is_negative, t.phi,  tbar.phi),   # had_top phi
            sel(lep_is_negative, t.mass, tbar.mass),  # had_top mass
            [dr_tt],           # ΔR(t, t̄)
            [dr_lep_had],      # ΔR(l, had_top)
            [m_ttbar],         # M(tt̄)
            [cos_theta_star],  # cos θ* (production angle in ttbar CM frame)
            # ── Tier 2: tt̄ system kinematics ──
            [dy_tt],           # Δy(t, t̄)
            [dphi_tt],         # Δφ(t, t̄)
            [pt_tt],           # pT(tt̄)
            [y_tt],            # y(tt̄)
            # ── Tier 1: spin correlation / polarization (helicity frame) ──
            [cos_theta_l],     # cosθ*_l   — leptonic-top analyzer = charged lepton
            [cos_theta_had],   # cosθ*_had — hadronic-top analyzer = down-type quark
            [dphi_l_had],      # Δφ(l, had_top), lab
            # ── 3D spin-basis projections (common {n,r,k}; building blocks of B/C) ──
            [cos_lep_n],       # cosθ_n^lep
            [cos_lep_r],       # cosθ_r^lep
            [cos_lep_k],       # cosθ_k^lep
            [cos_had_n],       # cosθ_n^had
            [cos_had_r],       # cosθ_r^had
            [cos_had_k],       # cosθ_k^had
            # ── opening-angle / relative-velocity combinations ──
            [beta_t_star],     # β* of leptonic top in hadronic-top rest frame
            [c_hel],           # c_hel = û_l·û_had opening angle  (⟨c_hel⟩ = −D/3)
            [c_han],           # n+r−k spin combination (Bell-type)
        ]

    def calc_pair_features(self, leps, jets):
        """Tier 3: pairwise lepton-jet / jet-jet geometry and masses.
        Computed on the leading 4 jets; combinatoric (no truth matching), so the
        same code transfers directly to reco-level jets. Returns a list of
        [1D numpy array] entries, one per feature, for np.concatenate splicing."""
        lep_eta  = leps.eta.to_numpy().astype(float)
        lep_phi  = leps.phi.to_numpy().astype(float)
        lep_pt   = leps.pt.to_numpy().astype(float)
        lep_mass = leps.mass.to_numpy().astype(float)
        lep_vec  = _to_cartesian(lep_pt, lep_eta, lep_phi, lep_mass)

        # leading-4 jet kinematics as numpy
        j_eta  = [jets.eta[:, i].to_numpy().astype(float)  for i in range(4)]
        j_phi  = [jets.phi[:, i].to_numpy().astype(float)  for i in range(4)]
        j_pt   = [jets.pt[:, i].to_numpy().astype(float)   for i in range(4)]
        j_mass = [jets.mass[:, i].to_numpy().astype(float) for i in range(4)]
        j_flav = [jets.hadronFlavour[:, i].to_numpy()      for i in range(4)]
        j_vec  = [_to_cartesian(j_pt[i], j_eta[i], j_phi[i], j_mass[i]) for i in range(4)]

        # ΔR(lepton, jet_i)
        dr_l_j = [_delta_r(lep_eta, lep_phi, j_eta[i], j_phi[i]) for i in range(4)]

        # jet-pair ΔR and invariant mass (6 unordered pairs of the leading 4 jets)
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        dr_jj = [_delta_r(j_eta[i], j_phi[i], j_eta[j], j_phi[j]) for (i, j) in pairs]
        m_jj  = [_pair_mass(j_vec[i], j_vec[j]) for (i, j) in pairs]

        # m(l, b): min over b-tagged leading jets (hadronFlavour == 5); 0 if none
        m_lb = np.stack([_pair_mass(lep_vec, j_vec[i]) for i in range(4)])   # (4, N)
        is_b = np.stack([j_flav[i] == 5 for i in range(4)])                  # (4, N)
        m_lb_min = np.min(np.where(is_b, m_lb, np.inf), axis=0)
        m_lb_min = np.where(np.isfinite(m_lb_min), m_lb_min, 0.)

        return (
            [[dr_l_j[i]] for i in range(4)] +   # ΔR(l, jet0..3)
            [[x] for x in dr_jj] +              # ΔR(jet_i, jet_j), 6 pairs
            [[x] for x in m_jj] +               # m(jet_i, jet_j), 6 pairs
            [[m_lb_min]]                        # min m(l, b)
        )

    def postprocess(self, accumulator):
        return accumulator
