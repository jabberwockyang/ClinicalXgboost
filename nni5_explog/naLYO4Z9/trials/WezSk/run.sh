#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='naLYO4Z9'
export NNI_SYS_DIR='/root/ClinicalXgboost/nni5_explog/naLYO4Z9/trials/WezSk'
export NNI_TRIAL_JOB_ID='WezSk'
export NNI_OUTPUT_DIR='/root/ClinicalXgboost/nni5_explog/naLYO4Z9/trials/WezSk'
export NNI_TRIAL_SEQ_ID='8'
export NNI_CODE_DIR='/root/ClinicalXgboost'
cd $NNI_CODE_DIR
eval python3 train_nni.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --exp_dir nni5_explog --groupingparams groupingsetting.yml 1>/root/ClinicalXgboost/nni5_explog/naLYO4Z9/trials/WezSk/stdout 2>/root/ClinicalXgboost/nni5_explog/naLYO4Z9/trials/WezSk/stderr
echo $? `date +%s%3N` >'/root/ClinicalXgboost/nni5_explog/naLYO4Z9/trials/WezSk/.nni/state'