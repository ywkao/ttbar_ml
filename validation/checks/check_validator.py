import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
print(f"check: {PROJECT_ROOT}")

from validation import nanoAOD_loader, tensor_loader, validator, utils
import os

import numpy as np
from validation.schema import WC_NAMES, OPERATORS, SM_NAME
from validation.binning import get_edges


nano_file = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
tensor_file = "/eos/uscms/store/user/ywkao/Outputs_sbi/pretraining/train.p"

run_features = False
run_weights = True

nano = nanoAOD_loader.load([nano_file], nevents=20000)

if run_features:
    tens = tensor_loader.load([tensor_file])
    res = validator.compare_features(nano, tens)

    for name in ["lep_pt", "njets", "cos_theta_star"]:
        print(f"{name:16s} max|Δ| = {res[name]['max_abs_diff']:.4g}")

    os.makedirs("output/feature_veriry", exist_ok=True)
    for name, r in res.items():
        utils.overlay(name, r["edges"], r["Na"], r["Nb"], outpath=f"output/feature_veriry/{name}.png")
    print("done, 41 feature plots")

if run_weights:
    # validate weights
    w = nanoAOD_loader.load_weights([nano_file], nevents=20000)
    res_w = validator.compare_weights(w["direct"], w["poly"])
    for name in WC_NAMES:
        r = res_w[name]
        flag = "" if r["corr"] > 0.9999 else "  <-- 檢查"
        print(f"{name:10s} corr={r['corr']:.6f}  max|Δ|={r['max_abs_diff']:.3g}{flag}")

    # ── (A) 驗證圖:每個 WC 點一張 direct-vs-poly scatter ──
    os.makedirs("output/weight_verify", exist_ok=True)
    for name in WC_NAMES:
        utils.weight_scatter(
            name, w["direct"][name], w["poly"][name],
            corr=res_w[name]["corr"],
            outpath=f"output/weight_verify/{name}.png",
        )
    print("done, 17 scatter plots (A)")
    
    # ── (B) 物理圖:每個 feature 一張 SM+16 EFT overlay ──
    # 用 poly weight(已驗證 corr=1 可信)畫 EFT 效應。注意:不 density,要看 rate+shape。
    os.makedirs("output/eft_overlay", exist_ok=True)
    for fname in nano["features"]:
        edges = get_edges(fname)
        vals = nano["features"][fname]
        N_sm, _ = np.histogram(vals, bins=edges, weights=w["direct"][SM_NAME])
        N_bsm = {
            op: np.histogram(vals, bins=edges, weights=w["direct"][op])[0]
            for op in OPERATORS
        }
        utils.overlay_multi(fname, edges, N_sm, N_bsm, outpath=f"output/eft_overlay/{fname}.png")
    print("done, 41 EFT overlay plots (B)")
