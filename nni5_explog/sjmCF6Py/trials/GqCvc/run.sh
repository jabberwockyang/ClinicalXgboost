#!/bin/bash
export NNI_PLATFORM='local'
export NNI_EXP_ID='sjmCF6Py'
export NNI_SYS_DIR='/root/ClinicalXgboost/nni5_explog/sjmCF6Py/trials/GqCvc'
export NNI_TRIAL_JOB_ID='GqCvc'
export NNI_OUTPUT_DIR='/root/ClinicalXgboost/nni5_explog/sjmCF6Py/trials/GqCvc'
export NNI_TRIAL_SEQ_ID='1'
export NNI_CODE_DIR='/root/ClinicalXgboost'
cd $NNI_CODE_DIR
eval python3 train_nni.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --exp_dir nni5_explog --groupingparams groupingsetting.yml 1>/root/ClinicalXgboost/nni5_explog/sjmCF6Py/trials/GqCvc/stdout 2>/root/ClinicalXgboost/nni5_explog/sjmCF6Py/trials/GqCvc/stderr
echo $? `date +%s%3N` >'/root/ClinicalXgboost/nni5_explog/sjmCF6Py/trials/GqCvc/.nni/state'