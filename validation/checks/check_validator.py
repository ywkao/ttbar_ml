import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
print(f"check: {PROJECT_ROOT}")

import os
import csv
import numpy as np
from validation import nanoAOD_loader, tensor_loader, validator, utils
from validation.schema import WC_NAMES, OPERATORS, SM_NAME
from validation.binning import get_edges

run_features = True
run_nano_weights = True
run_tens_weights = True
normalized = True

run_features = False
run_nano_weights = False
run_tens_weights = True
normalized = False

run_features = True
run_nano_weights = True
run_tens_weights = True
normalized = False

nano_file = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
tensor_file = "/eos/uscms/store/user/ywkao/Outputs_sbi/pretraining/train.p"

if run_features or run_nano_weights:
    nano = nanoAOD_loader.load([nano_file], with_weights=True, nevents=20000)

if run_features or run_tens_weights:
    tens = tensor_loader.load([tensor_file], with_weights=True)

#--------------------------------------------------

if run_features:
    res = validator.compare_features(nano, tens)

    for name in ["lep_pt", "njets", "cos_theta_star"]:
        print(f"{name:16s} max|Δ| = {res[name]['max_abs_diff']:.4g}")

    os.makedirs("output/feature_verify", exist_ok=True)
    for name, r in res.items():
        utils.overlay(name, r["edges"], r["Na"], r["Nb"], outpath=f"output/feature_verify/{name}.png")
    print("done, 72 feature plots")

#--------------------------------------------------

if run_tens_weights:
    rows = validator.shape_metrics(tens)            # 算
    utils.write_sensitivity_csv(rows, "output/sensitivity.csv")  # 寫檔(呈現)
    utils.print_top_n(rows, n=20)                   # 印(呈現)
    utils.print_best_per_op(rows)                   # 印(呈現)


if run_tens_weights:
    outdir = "output/eft_overlay_tensor_norm" if normalized else "output/eft_overlay_tensor"
    ylabel_ratio = "(normalized EFT)/(normalized SM)" if normalized else r"$c_i = 1$ / SM"
    os.makedirs(outdir, exist_ok=True)

    for idx, fname in enumerate(tens["features"]):
        edges = get_edges(fname)
        vals = tens["features"][fname]
        N_sm, _ = np.histogram(vals, bins=edges, weights=tens["weights"][SM_NAME])
        N_bsm = {
            op: np.histogram(vals, bins=edges, weights=tens["weights"][op])[0]
            for op in OPERATORS
        }
        if normalized:
            N_sm, N_bsm = utils.normalize_hists(N_sm, N_bsm, edges)
        utils.overlay_multi(fname, edges, N_sm, N_bsm, outpath=f"{outdir}/{fname}.png", ylabel_ratio=ylabel_ratio)

#--------------------------------------------------

if run_nano_weights:
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
    print("done, 72 EFT overlay plots (B)")
