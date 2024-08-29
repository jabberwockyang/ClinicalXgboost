# nni and determine top variables with importance

conda activate nni
cd /root/project_240828
export LD_LIBRARY_PATH=/root/anaconda3/envs/transformer/lib:$LD_LIBRARY_PATH
nnictl create --config config_nni1.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
python3 importance.py --nnidir nni1_explog/ZpoUyrIC --metric default --minimize True --number_of_trials 7
nnictl resume ZpoUyrIC --port 8081

# nni with derived variables
conda activate nni
cd /root/project_240828
export LD_LIBRARY_PATH=/root/anaconda3/envs/transformer/lib:$LD_LIBRARY_PATH
nnictl create --config config_dv.yml --port 8081
python3 importance.py --nnidir nni2_explog/_latest --metric default --minimize True --number_of_trials 7


# train with grouped data in top 7 params
conda activate nni
cd /root/project_240828
export LD_LIBRARY_PATH=/root/anaconda3/envs/transformer/lib:$LD_LIBRARY_PATH
python3 train_grouping.py --config grouping.yaml



nnictl stop -a









nnictl resume 4wkoauh9