# nni and determine top variables with importance
conda activate nni
cd /root/ClinicalXgboost

nnictl create --config config_nni1.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
python3 importance.py --nnidir nni1_explog/ZpoUyrIC --metric default --minimize True --number_of_trials 7

# nni with derived variables
nnictl create --config config_dv.yml --port 8081 # 3eQbjfcG  nnictl resume 3eQbjfcG --port 8081
python3 importance.py --nnidir nni2_explog/3eQbjfcG --metric default --minimize True --number_of_trials 7

# train with grouped data in top 7 params
python3 train_grouping.py --config grouping_nni2_default_top7.yaml
python3 train_grouping.py --config grouping_nni2_default_top25.yaml
python3 train_grouping.py --config grouping_nni2_roc_top25.yaml
python3 train_grouping.py --config grouping_nni1_default_top7.yaml
python3 train_grouping.py --config grouping_nni1_default_top25.yaml
python3 train_grouping.py --config grouping_nni1_roc_top25.yaml



for folder in gr_explog/*; do python3 importance.py --repodir $folder; done









nnictl stop -a
nnictl resume 4wkoauh9