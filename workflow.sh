# nni and determine top variables with importance
conda activate nni
cd /root/ClinicalXgboost

nnictl create --config config_nni1.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
python3 importance.py --nnidir nni1_explog/ZpoUyrIC --metric default --minimize True --number_of_trials 7

# nni with derived variables
nnictl create --config config_dv.yml --port 8081
python3 importance.py --nnidir nni2_explog/3eQbjfcG --metric default --minimize True --number_of_trials 7

# train with grouped data in top 7 params
python3 train_grouping.py --config grouping.yaml

python3 importance.py --repodir /root/ClinicalXgboost/gr_explog/3eQbjfcG_default_top7
python3 plt_roc_summary.py









nnictl stop -a
nnictl resume 4wkoauh9