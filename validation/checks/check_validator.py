import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
from validation import nanoAOD_loader, tensor_loader, validator, utils
from validation.schema import WC_NAMES, OPERATORS, SM_NAME, FEATURE_NAMES
from validation.binning import get_edges


DEFAULT_NANO   = "/eos/uscms/store/user/honor/TTbarSemileptonic/modCentral/251114_001833/0000/nanogen_modCentral_1.root"
DEFAULT_TENSOR = "/eos/uscms/store/user/ywkao/Outputs_sbi/pretraining/train.p"
DEFAULT_OUTPUT = "sample_validation"


def parse_args():
    p = argparse.ArgumentParser(
        description="Validator checks: feature & EFT weight comparisons",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── which blocks to run ──
    g = p.add_argument_group("blocks")
    g.add_argument("--features",     action="store_true",
                   help="compare nano vs tensor feature histograms")
    g.add_argument("--tens-weights", action="store_true",
                   help="tensor EFT sensitivity: shape metrics + CSV + tables")
    g.add_argument("--eft-overlay",  action="store_true",
                   help="EFT overlay plots from tensor weights")
    g.add_argument("--nano-weights", action="store_true",
                   help="validate direct vs poly weights + EFT overlay from nano")

    # ── table options (--tens-weights) ──
    g = p.add_argument_group("table options")
    g.add_argument("--metric", default="chi2_shape",
                   choices=["chi2_shape", "kl_div", "rate_change"],
                   help="metric shown in the feature-WC table")
    g.add_argument("--top-n",  type=int, default=10,
                   help="number of feature rows in the table")
    g.add_argument("--latex",   action="store_true",
                   help="print feature-WC table as LaTeX tabular")
    g.add_argument("--heatmap", action="store_true",
                   help="2D sensitivity heatmap (y=WC, x=per-WC rank, z=chi2/Σchi2)")
    g.add_argument("--wc-value", type=float, default=1.0,
                   help="operator coefficient used to reconstruct tensor-side "
                        "BSM weights (SM term is always cSM=1); affects "
                        "--tens-weights/--heatmap/--eft-overlay, not --nano-weights")

    # ── overlay options (--eft-overlay / --nano-weights) ──
    g = p.add_argument_group("overlay options")
    g.add_argument("--normalized", action="store_true",
                   help="normalize EFT histograms when making overlay plots")

    # ── file / path options ──
    g = p.add_argument_group("files")
    g.add_argument("--nano-file",   default=DEFAULT_NANO,   metavar="PATH")
    g.add_argument("--tensor-file", default=DEFAULT_TENSOR, metavar="PATH")
    g.add_argument("--output",      default=DEFAULT_OUTPUT, metavar="DIR")
    g.add_argument("--nevents",     type=int, default=20000,
                   help="max events loaded from NanoAOD")

    return p.parse_args()


def main():
    args = parse_args()
    output = args.output

    need_nano   = args.features or args.nano_weights
    need_tensor = args.features or args.tens_weights or args.eft_overlay or args.heatmap

    if need_nano:
        nano = nanoAOD_loader.load([args.nano_file], with_weights=True, nevents=args.nevents)

    if need_tensor:
        tens = tensor_loader.load([args.tensor_file], with_weights=True,
                                   wc_value=args.wc_value)

    # suffix tensor-weight-derived output names when wc_value deviates from
    # the nano-side reference point (1.0), so runs don't clobber each other
    wc_suffix = "" if args.wc_value == 1.0 else f"_wc{args.wc_value:g}"

    # ── Block 1: feature histograms — nano vs tensor ──
    if args.features:
        res = validator.compare_features(nano, tens)
        for name in ["lep_pt", "njets", "cos_theta_star"]:
            print(f"{name:16s} max|Δ| = {res[name]['max_abs_diff']:.4g}")
        os.makedirs(f"{output}/feature_verify", exist_ok=True)
        for name, r in res.items():
            utils.overlay(name, r["edges"], r["Na"], r["Nb"],
                          outpath=f"{output}/feature_verify/{name}.png")
        print(f"done, {len(FEATURE_NAMES)} feature plots")

    # ── Block 2: EFT sensitivity from tensor weights ──
    if args.tens_weights or args.heatmap:
        rows = validator.shape_metrics(tens)
        os.makedirs(output, exist_ok=True)
    if args.tens_weights:
        utils.write_sensitivity_csv(rows, f"{output}/sensitivity{wc_suffix}.csv")
        utils.print_top_n(rows, n=20)
        utils.print_best_per_op(rows)
        utils.print_feature_wc_table(rows, n=args.top_n, metric=args.metric, latex=args.latex)
    if args.heatmap:
        heatmap_path = f"{output}/sensitivity_heatmap{wc_suffix}.png"
        utils.plot_sensitivity_heatmap(
            rows, n=args.top_n, metric=args.metric,
            outpath=heatmap_path,
        )
        print(f"done, sensitivity heatmap → {heatmap_path}")

    # ── Block 3: EFT overlay plots from tensor ──
    if args.eft_overlay:
        outdir = (f"{output}/eft_overlay_tensor_norm" if args.normalized
                  else f"{output}/eft_overlay_tensor")
        ylabel_ratio = ("(normalized EFT)/(normalized SM)" if args.normalized
                        else rf"$c_i = {args.wc_value:g}$ / SM")
        os.makedirs(outdir, exist_ok=True)
        for fname in tens["features"]:
            edges = get_edges(fname)
            vals  = tens["features"][fname]
            N_sm, _ = np.histogram(vals, bins=edges, weights=tens["weights"][SM_NAME])
            N_bsm = {
                op: np.histogram(vals, bins=edges, weights=tens["weights"][op])[0]
                for op in OPERATORS
            }
            if args.normalized:
                N_sm, N_bsm = utils.normalize_hists(N_sm, N_bsm, edges)
            suffix = wc_suffix if wc_suffix else "_wc1"
            utils.overlay_multi(fname, edges, N_sm, N_bsm,
                                outpath=f"{outdir}/{fname}{suffix}.png", ylabel_ratio=ylabel_ratio)

    # ── Block 4: nano weight validation (direct vs poly) + EFT overlay ──
    if args.nano_weights:
        w     = nanoAOD_loader.load_weights([args.nano_file], nevents=args.nevents)
        res_w = validator.compare_weights(w["direct"], w["poly"])
        for name in WC_NAMES:
            r    = res_w[name]
            flag = "" if r["corr"] > 0.9999 else "  <-- check"
            print(f"{name:10s} corr={r['corr']:.6f}  max|Δ|={r['max_abs_diff']:.3g}{flag}")

        os.makedirs(f"{output}/weight_verify", exist_ok=True)
        for name in WC_NAMES:
            utils.weight_scatter(
                name, w["direct"][name], w["poly"][name],
                corr=res_w[name]["corr"],
                outpath=f"{output}/weight_verify/{name}.png",
            )
        print("done, 17 scatter plots (A)")

        os.makedirs(f"{output}/eft_overlay", exist_ok=True)
        for fname in nano["features"]:
            edges = get_edges(fname)
            vals  = nano["features"][fname]
            N_sm, _ = np.histogram(vals, bins=edges, weights=w["direct"][SM_NAME])
            N_bsm = {
                op: np.histogram(vals, bins=edges, weights=w["direct"][op])[0]
                for op in OPERATORS
            }
            utils.overlay_multi(fname, edges, N_sm, N_bsm,
                                outpath=f"{output}/eft_overlay/{fname}.png")
        print(f"done, {len(FEATURE_NAMES)} EFT overlay plots (B)")


if __name__ == "__main__":
    main()
