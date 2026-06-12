#!/bin/bash
# Setup script for topsbi_condor: a minimal conda env for condor batch jobs.
#
# Purpose: run run_pretraining_processor.py (sbi mode) on cmslpc condor worker nodes,
#          where NFS (/uscms_data) is not mounted. All files are fetched from EOS.
#
# Key packages (everything else is auto-pulled as dependencies):
#   - python 3.11
#   - coffea 2026.5.0      (NanoAOD processing via coffea.processor)
#   - fsspec-xrootd 0.5.2  (required for uproot to open root:// xrootd URIs)
#   - torch 2.12.0 CPU     (tensor save/load; CPU-only, no CUDA needed for pretraining)
#   - conda-pack 0.9.1     (for packaging the env into a portable tarball)
#
# Usage: run this script once on an interactive node to (re)create the env.
# After running, pack and upload to EOS with the commands at the bottom.

CONDA=/uscms_data/d3/ywkao/Projects/topsbi/miniforge3/bin/conda
EOS=root://cmseos.fnal.gov//store/user/ywkao

# ── 1. Create base env ────────────────────────────────────────────────────────
$CONDA create -n topsbi_condor python=3.11 -y

# ── 2. Install coffea (brings in awkward, uproot, numpy, matplotlib, etc.) ───
$CONDA install -n topsbi_condor -c conda-forge coffea=2026.5.0 -y

# ── 3. Install fsspec-xrootd (needed for coffea/uproot to read root:// URIs) ─
$CONDA install -n topsbi_condor -c conda-forge fsspec-xrootd -y

# ── 4. Install torch CPU-only (pretraining jobs don't need GPU) ──────────────
$CONDA run -n topsbi_condor pip install torch --index-url https://download.pytorch.org/whl/cpu

# ── 5. Install conda-pack (for packaging) ────────────────────────────────────
$CONDA install -n topsbi_condor -c conda-forge conda-pack -y

# ── 6. Pack and upload to EOS ─────────────────────────────────────────────────
# Run after confirming all packages are present:
#   $CONDA list -n topsbi_condor | grep -E "coffea|fsspec-xrootd|torch|conda-pack"
#
cd /tmp
$CONDA run -n topsbi_condor \
    conda-pack -n topsbi_condor -o topsbi_condor_env.tar.gz --ignore-missing-files
#
# Note: --ignore-missing-files is needed because pip-installed torch overwrites
#       some conda-managed setuptools files, causing a conflict warning.
#
xrdfs ${EOS} mkdir -p /store/user/ywkao/envs
xrdcp /tmp/topsbi_condor_env.tar.gz ${EOS}/envs/topsbi_condor_env.tar.gz
rm /tmp/topsbi_condor_env.tar.gz

# ── EOS layout expected by run_batch.sh ───────────────────────────────────────
# ${EOS}/envs/topsbi_condor_env.tar.gz   <- packed conda env (this script)
# ${EOS}/code/ttbar_ml_code.tar.gz       <- code tarball (see below)
# ${EOS}/Outputs_sbi/pretraining/        <- job outputs written here
#
# Code tarball contains:
#   run_pretraining_processor.py
#   ttbar_utilities/  utils/  pretraining/
#   Inputs/pretraining/TT01j1l_modCentral_updated_ywk.json
#   Inputs/rwgt_card_modCentral.dat
#
# Rebuild code tarball after any code change:
#   cd /uscms_data/d3/ywkao/Projects/ttbar_ml
#   tar -czf /tmp/ttbar_ml_code.tar.gz \
#       run_pretraining_processor.py \
#       ttbar_utilities/ utils/ pretraining/ \
#       Inputs/pretraining/TT01j1l_modCentral_updated_ywk.json \
#       Inputs/rwgt_card_modCentral.dat
#   xrdfs ${EOS} mkdir -p /store/user/ywkao/code
#   xrdcp /tmp/ttbar_ml_code.tar.gz ${EOS}/code/ttbar_ml_code.tar.gz
#   rm /tmp/ttbar_ml_code.tar.gz
