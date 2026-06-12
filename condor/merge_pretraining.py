#!/usr/bin/env python
"""Merge batch_*.p files produced by run_pretraining_processor.py --batch-id,
then do a single train/test/val split and save the final datasets."""

import argparse
import os
import glob
import torch
from torch import save, seed
from torch.utils.data import TensorDataset, random_split


def main():
    parser = argparse.ArgumentParser(description='Merge pretraining batches into train/test/val splits')
    parser.add_argument('outpath', help='Directory containing batch_*.p files (also where output is written)')
    parser.add_argument('--split', nargs=3, type=float, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'TEST', 'VAL'), help='Train/test/val fractions (default: 0.8 0.1 0.1)')
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
    print(f"Total events: {features.shape[0]}, features shape: {features.shape}, coefs shape: {coefs.shape}")

    train, test, val = random_split(TensorDataset(features, coefs), args.split)

    print(f"\nSplitting: train={len(train)}, test={len(test)}, val={len(val)}")
    print(f"Saving to {args.outpath} ...")
    save(TensorDataset(train[:][0], train[:][1]), os.path.join(args.outpath, "train.p"))
    save(TensorDataset(test[:][0],  test[:][1]),  os.path.join(args.outpath, "test.p"))
    save(TensorDataset(val[:][0],   val[:][1]),   os.path.join(args.outpath, "validation.p"))
    print("Done!")


if __name__ == '__main__':
    main()
