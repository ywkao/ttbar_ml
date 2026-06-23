"""
Validation: compare direct LHEWeight from NanoAOD vs polynomial reconstruction from fit_coefs.

If the two agree, train.p is correct and any "similar distributions" reflects true physics.
If they disagree, there is a bug in the polynomial fit or reconstruction.
"""
import re
import numpy as np
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

NanoAODSchema.warn_missing_crossrefs = False

FILE = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
N_EVENTS = 2000   # increase for more statistics
OUTDIR   = "plots_validation"

import os; os.makedirs(OUTDIR, exist_ok=True)

# ── Load events ───────────────────────────────────────────────────────────────
print(f"Loading {N_EVENTS} events from {FILE}")
events = NanoEventsFactory.from_root(
    {FILE: 'Events'}, schemaclass=NanoAODSchema
).events()[:N_EVENTS]

# ── Find ctGRe=1 single-operator LHEWeight field ─────────────────────────────
ctgre_field = None
for fname in events.LHEWeight.fields:
    m = re.match(r'^EFTrwgt(\d+)_(.*)$', fname)
    if not m:
        continue
    n = int(m.group(1))
    if not (201 <= n <= 216):
        continue
    tokens = m.group(2).split('_')
    for i in range(0, len(tokens) - 1, 2):
        op, val = tokens[i], tokens[i + 1]
        if op == 'ctGRe' and val == '1p':
            ctgre_field = fname
            break
    if ctgre_field:
        break

if ctgre_field is None:
    raise RuntimeError("Could not find ctGRe=1 single-operator LHEWeight field")
print(f"ctGRe=1 field: {ctgre_field}")

# ── Apply the same event selection as ml_data.py ─────────────────────────────
import pretraining.ml_data as ml_data
from pretraining.analysis_tools import genObjectSelection, genEventSelection

cleanleps, cleanjets = genObjectSelection(events)
mask = genEventSelection(cleanleps, cleanjets)
print(f"Selected {int(ak.sum(mask))} / {N_EVENTS} events")

# ── Direct weights from NanoAOD ───────────────────────────────────────────────
w_sm_direct  = ak.to_numpy(events.LHEWeight['sm_point'][mask]).astype(float)
w_eft_direct = ak.to_numpy(events.LHEWeight[ctgre_field][mask]).astype(float)

# ── Polynomial reconstruction from fit_coefs ─────────────────────────────────
from pretraining.fitCoefficients import calculate_fit, vec, wcs as FIT_WCS

n_wc = len(FIT_WCS)
fields = events.LHEWeight.fields
eft_coeffs, _ = calculate_fit(fields, events.LHEWeight[mask])   # (n_pairs, n_sel)
coefs = eft_coeffs.T                                             # (n_sel, n_pairs)

# Build WC vectors using the CORRECT vec convention (with sqrt(2))
c0 = np.zeros(n_wc); c0[FIT_WCS.index('cSM')]   = 1.0
c1 = np.zeros(n_wc); c1[FIT_WCS.index('cSM')]   = 1.0
c1[FIT_WCS.index('ctGRe')] = 1.0

w_sm_poly  = coefs @ vec(np.outer(c0, c0))
w_eft_poly = coefs @ vec(np.outer(c1, c1))

# ── Print comparison ──────────────────────────────────────────────────────────
n = len(w_sm_direct)
corr_sm  = np.corrcoef(w_sm_direct,  w_sm_poly)[0, 1]
corr_eft = np.corrcoef(w_eft_direct, w_eft_poly)[0, 1]

print(f"\n{'':30s}  {'direct':>12s}  {'poly':>12s}  {'corr':>8s}")
print(f"{'w(SM)  mean ± std':30s}  {w_sm_direct.mean():>8.4f}±{w_sm_direct.std():<6.4f}"
      f"  {w_sm_poly.mean():>8.4f}±{w_sm_poly.std():<6.4f}  {corr_sm:>8.4f}")
print(f"{'w(EFT) mean ± std':30s}  {w_eft_direct.mean():>8.4f}±{w_eft_direct.std():<6.4f}"
      f"  {w_eft_poly.mean():>8.4f}±{w_eft_poly.std():<6.4f}  {corr_eft:>8.4f}")

ratio_direct = w_eft_direct / np.where(w_sm_direct > 0, w_sm_direct, np.nan)
ratio_poly   = w_eft_poly   / np.where(w_sm_poly   > 0, w_sm_poly,   np.nan)
print(f"\nEFT/SM ratio (direct): median={np.nanmedian(ratio_direct):.4f}  std={np.nanstd(ratio_direct):.4f}")
print(f"EFT/SM ratio (poly):   median={np.nanmedian(ratio_poly):.4f}  std={np.nanstd(ratio_poly):.4f}")

# ── Feature arrays (same as ml_data.py ordering) ─────────────────────────────
leps = ak.flatten(cleanleps[mask])
jets = cleanjets[mask]
met  = events.GenMET[mask]

FEATURE_NAMES = [
    'lep_pt','lep_eta','lep_phi','lep_mass',
    'met_pt','met_phi',
    'jet0_pt','jet0_eta','jet0_phi','jet0_mass',
    'jet1_pt','jet1_eta','jet1_phi','jet1_mass',
    'jet2_pt','jet2_eta','jet2_phi','jet2_mass',
    'jet3_pt','jet3_eta','jet3_phi','jet3_mass',
    'njets','HT',
    'jet0_flav','jet1_flav','jet2_flav','jet3_flav',
    'lep_top_pt','lep_top_eta','lep_top_phi','lep_top_mass',
    'had_top_pt','had_top_eta','had_top_phi','had_top_mass',
]

gp = events.GenPart[mask]
is_final = gp.hasFlags(["fromHardProcess", "isLastCopy"])
t    = ak.fill_none(ak.pad_none(gp[is_final & (gp.pdgId ==  6)], 1)[:, 0].pt, 0.)
tbar = ak.fill_none(ak.pad_none(gp[is_final & (gp.pdgId == -6)], 1)[:, 0].pt, 0.)

feat_arrays = {
    'lep_pt':      ak.to_numpy(leps.pt).astype(float),
    'lep_eta':     ak.to_numpy(leps.eta).astype(float),
    'met_pt':      ak.to_numpy(met.pt).astype(float),
    'lep_top_pt':  ak.to_numpy(t).astype(float),
    'had_top_pt':  ak.to_numpy(tbar).astype(float),
    'HT':          ak.to_numpy(ak.sum(jets.pt, axis=1)).astype(float),
}

# ── Overlay plots: direct vs poly, for SM and EFT ────────────────────────────
BINNING = {
    'lep_pt':      (40, 0, 400),
    'lep_eta':     (40,-3, 3),
    'met_pt':      (40, 0, 400),
    'lep_top_pt':  (40, 0, 700),
    'had_top_pt':  (40, 0, 700),
    'HT':          (40, 0,2000),
}

for fname, vals in feat_arrays.items():
    nb, lo, hi = BINNING[fname]
    edges = np.linspace(lo, hi, nb + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7),
                                    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05},
                                    sharex=True)

    def hw(ax, vals, w, **kw):
        h, _ = np.histogram(vals, bins=edges, weights=w)
        s = h.sum()
        if s > 0: h = h / s
        ax.stairs(h, edges, **kw)
        return h

    h_sm_d  = hw(ax1, vals, w_sm_direct,  color='black',  lw=2,   label='SM (direct)')
    h_eft_d = hw(ax1, vals, w_eft_direct, color='crimson',lw=2,   label='EFT direct (ctGRe=1)')
    h_sm_p  = hw(ax1, vals, w_sm_poly,    color='black',  lw=1.5, ls='--', label='SM (poly)')
    h_eft_p = hw(ax1, vals, w_eft_poly,   color='crimson',lw=1.5, ls='--', label='EFT poly (ctGRe=1)')

    ax1.set_ylabel('shape (normalised)')
    ax1.legend(fontsize=10, ncol=2)
    ax1.set_title(fname)

    with np.errstate(divide='ignore', invalid='ignore'):
        r_d = np.where(h_sm_d > 0, h_eft_d / h_sm_d, np.nan)
        r_p = np.where(h_sm_p > 0, h_eft_p / h_sm_p, np.nan)

    ax2.stairs(r_d, edges, color='crimson', lw=2,   label='direct')
    ax2.stairs(r_p, edges, color='crimson', lw=1.5, ls='--', label='poly')
    ax2.axhline(1, color='grey', lw=1)
    ax2.set_ylabel('EFT / SM')
    ax2.set_ylim(0, 3)
    ax2.set_xlabel(fname)
    ax2.legend(fontsize=9)

    out_path = os.path.join(OUTDIR, f'{fname}_validation.png')
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {out_path}")

print("\nDone.")
