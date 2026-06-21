from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import pretraining.ml_data as ml_data
import numpy as np
import torch

import uproot
NEW_FILE = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
f = uproot.open(NEW_FILE)
all_branches = f["Events"].keys()
lhe_branches = [b for b in all_branches if b.startswith("LHEWeight_")]
print(f"Total LHEWeight branches in new file: {len(lhe_branches)}")
ctgre_branches = [b for b in lhe_branches if "ctGRe" in b]
print(f"LHEWeight branches containing 'ctGRe': {len(ctgre_branches)}")
# Look for single-operator ctGRe=1 point (ctGRe_1p with all others at 0p)
ctgre_1p = [b for b in ctgre_branches if "ctGRe_1p" in b]
print(f"Branches with ctGRe_1p: {len(ctgre_1p)}")
for b in ctgre_1p[:3]:
    print(" ", b)

# ── 1. Run processor on a small slice of events ──────────────────────────────
events = NanoEventsFactory.from_root(
    {'/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root': 'Events'},
    schemaclass=NanoAODSchema
).events()

p = ml_data.SemiLepProcessor(runFit=True)
out = p.process(events[:500])
print('features shape:', out['features'].get().shape)
print('fit_coefs shape:', out['fit_coefs'].get().shape)

# ── 2. Confirm LHEWeight fields contain ctGRe reweighting points ──────────────
lhe_fields = [f for f in events.LHEWeight.fields if 'ctGRe' in f]
print(f'\nLHEWeight fields containing "ctGRe": {len(lhe_fields)}')
for f in lhe_fields[:5]:
    print(' ', f)

# ── 3. Diagnose fit_coefs from this run ──────────────────────────────────────
from pretraining.fitCoefficients import wcs as FIT_WCS
print('\nfitCoefficients wcs order:')
for idx, name in enumerate(FIT_WCS):
    print(f'  [{idx:2d}] {name}')

n_wc = len(FIT_WCS)
j_idx, i_idx = np.tril_indices(n_wc)

def find_k(name_a, name_b):
    """Flat index for the (name_a, name_b) coefficient pair."""
    a, b = FIT_WCS.index(name_a), FIT_WCS.index(name_b)
    lo, hi = min(a, b), max(a, b)
    hits = np.where((i_idx == lo) & (j_idx == hi))[0]
    return int(hits[0])

k_sm_sm    = find_k('cSM',   'cSM')
k_sm_ctgre = find_k('cSM',   'ctGRe')
k_ctgre_sq = find_k('ctGRe', 'ctGRe')

coefs = out['fit_coefs'].get().numpy()   # shape (n_events, n_wc_pairs)
print(f'\nCoefficient diagnostics (mean over {coefs.shape[0]} events):')
print(f'  SM²       [k={k_sm_sm:3d}]: {coefs[:, k_sm_sm].mean():.4f}')
print(f'  SM×ctGRe  [k={k_sm_ctgre:3d}]: {coefs[:, k_sm_ctgre].mean():.4f}')
print(f'  ctGRe²    [k={k_ctgre_sq:3d}]: {coefs[:, k_ctgre_sq].mean():.4f}')

# ── 4. Reconstruct w(SM) and w(ctGRe=1) and compare ─────────────────────────
def expand_array(c):
    """Lower-triangular outer product (no sqrt(2) scaling — matches expandArray)."""
    out = []
    for i in range(len(c)):
        for j in range(i + 1):
            out.append(c[i] * c[j])
    return np.array(out, dtype=np.float64)

c0 = np.zeros(n_wc); c0[FIT_WCS.index('cSM')]   = 1.0           # SM point
c1 = np.zeros(n_wc); c1[FIT_WCS.index('cSM')]   = 1.0           # EFT point
c1[FIT_WCS.index('ctGRe')] = 1.0

w_sm  = coefs @ expand_array(c0)
w_eft = coefs @ expand_array(c1)

print(f'\nReconstructed weight summary:')
print(f'  w(SM)       mean={w_sm.mean():.4f}  std={w_sm.std():.4f}')
print(f'  w(ctGRe=1)  mean={w_eft.mean():.4f}  std={w_eft.std():.4f}')
ratio = w_eft / np.where(w_sm > 0, w_sm, np.nan)
print(f'  ratio EFT/SM  mean={np.nanmean(ratio):.4f}  std={np.nanstd(ratio):.4f}')

# ── 5. Same diagnostics on the saved train.p ─────────────────────────────────
TRAIN_P = '/eos/uscms/store/user/ywkao/Outputs_sbi/pretraining/train.p'
try:
    _, saved_coefs = torch.load(TRAIN_P, weights_only=False)[:]
    saved_coefs = saved_coefs.numpy()
    print(f'\n── Saved train.p diagnostics ({saved_coefs.shape[0]} events) ──')
    print(f'  SM²       [k={k_sm_sm:3d}]: {saved_coefs[:, k_sm_sm].mean():.4f}')
    print(f'  SM×ctGRe  [k={k_sm_ctgre:3d}]: {saved_coefs[:, k_sm_ctgre].mean():.4f}')
    print(f'  ctGRe²    [k={k_ctgre_sq:3d}]: {saved_coefs[:, k_ctgre_sq].mean():.4f}')
    w_sm2  = saved_coefs @ expand_array(c0)
    w_eft2 = saved_coefs @ expand_array(c1)
    ratio2 = w_eft2 / np.where(w_sm2 > 0, w_sm2, np.nan)
    print(f'  ratio EFT/SM  mean={np.nanmean(ratio2):.4f}  std={np.nanstd(ratio2):.4f}')
except Exception as e:
    print(f'\n(Could not load {TRAIN_P}: {e})')
