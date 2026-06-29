#!/usr/bin/env python
"""Merge batch_*.p files produced by run_pretraining_processor.py --batch-id,
then do a single train/test/val split and save the final datasets."""

import argparse
import os
import glob
import gc
import torch
from torch.utils.data import TensorDataset


def main():
    parser = argparse.ArgumentParser(description='Merge pretraining batches into train/test/val splits')
    parser.add_argument('outpath', help='Directory containing batch_*.p files (also where output is written)')
    parser.add_argument('--split', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'TEST', 'VAL'), help='Train/test/val fractions')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the shuffle')
    args = parser.parse_args()

    batch_files = sorted(glob.glob(os.path.join(args.outpath, "batch_*.p")))
    if not batch_files:
        print(f"No batch_*.p files found in {args.outpath}")
        return 1
    print(f"Found {len(batch_files)} batch files")

    features_list, coefs_list = [], []
    for bf in batch_files:
        print(f"  Loading {bf} ...")
        ds = torch.load(bf, map_location="cpu", weights_only=False)
        features_list.append(ds.tensors[0])
        coefs_list.append(ds.tensors[1])

    features = torch.cat(features_list, dim=0)
    coefs    = torch.cat(coefs_list,    dim=0)
    del features_list, coefs_list
    gc.collect()
    print(f"Total events: {features.shape[0]}, features shape: {features.shape}, coefs shape: {coefs.shape}")

    # --- index-based split, no Subset / no per-element __getitem__ ---
    N = features.shape[0]
    f_train, f_test, f_val = args.split
    assert abs(f_train + f_test + f_val - 1.0) < 1e-6, "split fractions must sum to 1"

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)

    n_train = int(N * f_train)
    n_test  = int(N * f_test)
    n_val   = N - n_train - n_test
    print(f"\nSplitting: train={n_train}, test={n_test}, val={n_val}")
    print(f"Saving to {args.outpath} ...")

    splits = [
        ("train.p",      perm[:n_train]),
        ("test.p",       perm[n_train:n_train + n_test]),
        ("validation.p", perm[n_train + n_test:]),
    ]

    for fname, idx in splits:
        f_sub = features.index_select(0, idx)
        c_sub = coefs.index_select(0, idx)
        torch.save(TensorDataset(f_sub, c_sub), os.path.join(args.outpath, fname))
        print(f"  wrote {fname}  ({f_sub.shape[0]} events)")
        del f_sub, c_sub
        gc.collect()

    print("Done!")


if __name__ == '__main__':
    main()
