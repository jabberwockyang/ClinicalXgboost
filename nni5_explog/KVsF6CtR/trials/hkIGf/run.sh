#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='KVsF6CtR'
export NNI_SYS_DIR='/root/ClinicalXgboost/nni5_explog/KVsF6CtR/trials/hkIGf'
export NNI_TRIAL_JOB_ID='hkIGf'
export NNI_OUTPUT_DIR='/root/ClinicalXgboost/nni5_explog/KVsF6CtR/trials/hkIGf'
export NNI_TRIAL_SEQ_ID='12'
export NNI_CODE_DIR='/root/ClinicalXgboost'
cd $NNI_CODE_DIR
eval python3 train_nni.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --exp_dir nni5_explog --groupingparams groupingsetting.yml 1>/root/ClinicalXgboost/nni5_explog/KVsF6CtR/trials/hkIGf/stdout 2>/root/ClinicalXgboost/nni5_explog/KVsF6CtR/trials/hkIGf/stderr
echo $? `date +%s%3N` >'/root/ClinicalXgboost/nni5_explog/KVsF6CtR/trials/hkIGf/.nni/state'