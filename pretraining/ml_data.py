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

        return [
            sel(lep_is_negative, tbar.pt,   t.pt),    # lep_top pt
            sel(lep_is_negative, tbar.eta,  t.eta),   # lep_top eta
            sel(lep_is_negative, tbar.phi,  t.phi),   # lep_top phi
            sel(lep_is_negative, tbar.mass, t.mass),  # lep_top mass
            sel(lep_is_negative, t.pt,   tbar.pt),    # had_top pt
            sel(lep_is_negative, t.eta,  tbar.eta),   # had_top eta
            sel(lep_is_negative, t.phi,  tbar.phi),   # had_top phi
            sel(lep_is_negative, t.mass, tbar.mass),  # had_top mass
        ]

    def postprocess(self, accumulator):
        return accumulator
