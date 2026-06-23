import awkward as ak
import numpy as np
from torch import float64, tensor, from_numpy
from coffea.nanoevents import NanoAODSchema
from coffea.processor import ProcessorABC, defaultdict_accumulator

from .fitCoefficients import calculate_fit
from .analysis_tools import genObjectSelection, genEventSelection, topObjectSelection
from .TensorAccumulator import TensorAccumulator

NanoAODSchema.warn_missing_crossrefs = False

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
        ]

    def postprocess(self, accumulator):
        return accumulator
