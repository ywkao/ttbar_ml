# Prepare training samples using condor jobs

Step 1: Prepare tarballs for condor jobs
```bash
# python environment
sh setup_condor_env.sh

# upload code to EOS
sh upload_code.sh
```

Step 2: Submit condor jobs
```bash
condor_submit submit_pretraining.jdl
```

Step 3: merge training samples
```bash
python3 merge_pretraining.py /eos/uscms/store/user/ywkao/Outputs_sbi/pretraining/
```
