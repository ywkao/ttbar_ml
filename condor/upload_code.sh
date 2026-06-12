#!/bin/bash
# Run from anywhere after modifying code under ttbar_ml/.
# Repacks and uploads the code tarball to EOS.

PROJDIR=/uscms_data/d3/ywkao/Projects/ttbar_ml
EOS=root://cmseos.fnal.gov//store/user/ywkao
TARBALL=/tmp/ttbar_ml_code.tar.gz

cd ${PROJDIR}
tar -czf ${TARBALL} \
    run_pretraining_processor.py \
    ttbar_utilities/ utils/ pretraining/ \
    Inputs/pretraining/TT01j1l_modCentral_updated_ywk.json \
    Inputs/rwgt_card_modCentral.dat

xrdcp -f ${TARBALL} ${EOS}/code/ttbar_ml_code.tar.gz
rm ${TARBALL}
echo "Done: ttbar_ml_code.tar.gz uploaded to EOS"
