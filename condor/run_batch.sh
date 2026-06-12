#!/bin/bash
# Usage: run_batch.sh <batch_id> <file_start> <file_end>
BATCH_ID=$1
FILE_START=$2
FILE_END=$3

EOS=root://cmseos.fnal.gov//store/user/ywkao
EOS_ENV=${EOS}/envs/topsbi_condor_env.tar.gz
EOS_OUTPATH=${EOS}/Outputs_sbi/pretraining

echo "=== Batch ${BATCH_ID}: files ${FILE_START}-${FILE_END} ==="
echo "Host: $(hostname), Date: $(date)"

cd ${_CONDOR_SCRATCH_DIR}

# Fetch and unpack conda environment from EOS
echo "Fetching Python environment..."
xrdcp ${EOS_ENV} topsbi_condor_env.tar.gz || { echo "ERROR: xrdcp env failed"; exit 1; }
mkdir -p topsbi_condor_env
tar -xzf topsbi_condor_env.tar.gz -C topsbi_condor_env
PYTHON=${_CONDOR_SCRATCH_DIR}/topsbi_condor_env/bin/python3.11
${PYTHON} ${_CONDOR_SCRATCH_DIR}/topsbi_condor_env/bin/conda-unpack
export MPLCONFIGDIR=${_CONDOR_SCRATCH_DIR}

# Fetch and unpack code from EOS
echo "Fetching code..."
xrdcp ${EOS}/code/ttbar_ml_code.tar.gz ttbar_ml_code.tar.gz || { echo "ERROR: xrdcp code failed"; exit 1; }
tar -xzf ttbar_ml_code.tar.gz

OUTPATH=${_CONDOR_SCRATCH_DIR}/output
mkdir -p ${OUTPATH}

${PYTHON} run_pretraining_processor.py sbi \
    Inputs/pretraining/TT01j1l_modCentral_updated_ywk.json \
    -w Inputs/rwgt_card_modCentral.dat -f \
    -o ${OUTPATH} \
    --file-start ${FILE_START} \
    --file-end   ${FILE_END} \
    --batch-id   ${BATCH_ID}

EXIT_CODE=$?
echo "=== Batch ${BATCH_ID} finished with exit code ${EXIT_CODE} at $(date) ==="

# Transfer outputs to EOS
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Transferring outputs to EOS..."
    for f in ${OUTPATH}/*; do
        [ -f "$f" ] || continue
        xrdcp -f "$f" ${EOS_OUTPATH}/$(basename "$f") \
            && echo "  copied $(basename "$f")" \
            || { echo "ERROR: failed to copy $(basename "$f")"; EXIT_CODE=1; }
    done
fi

exit ${EXIT_CODE}
