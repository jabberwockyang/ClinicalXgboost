conda activate nni
cd /root/ClinicalXgboost


# nni and determine top variables with importance
nnictl create --config config_nni1.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
for expid in nni1_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
for expid in nni1_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 25; done

# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni1_*.yaml; do
    for expid in nni1_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni1_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done

# nni with derived variables
nnictl create --config config_dv.yml --port 8081 # 3eQbjfcG  nnictl resume 3eQbjfcG --port 8081
for expid in nni2_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
for expid in nni2_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 25; done
# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni2_*.yaml; do
    for expid in nni2_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni2_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done



# nni with derived variables and topn
nnictl create --config config_dv_topn.yml --port 8081 # 
for expid in nni3_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done
# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni3_*.yaml; do
    for expid in nni3_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni3_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done


# nni with topn
nnictl create --config config_topn.yml --port 7860
for expid in nni4_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done

# Loop through YAML files and run train_grouping.py for each experiment
for yml in grouping_nni4_*.yaml; do
    for expid in nni4_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni4_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done



# SyguB7Fb  n60M47dW topn 参数 50-200 
# 0T2GXABC  lcMYyo0V topn 参数 0.1-1


# to run importance for all gr_explog
for grfolder in gr_explog/*; do
    python3 importance.py --grdir "$grfolder"
done

# summary in variablesimportance shows no benefit of feature derivation and subsequent topn 
# use nni1 results for further analysis

# try different grouping strategies
for yml in grouping_nni1_*_gr2.yaml; do
    for expid in nni1_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

for expid in nni1_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*_gr2; do
        python3 importance.py --grdir "$grfolder"
    done
done

# results show the previous grouping strategy is better

# try boruta
python3 runboruta.py

#during boruta discover problem with imputation run nni1 again and grouping and importance

# BORUTA shows good performance in feature selection   

# a new workflow
# feature engingeering with minmxavg and acute chronic avg
###  done in sqlquery

# nni5 GET best params for boruta
nnictl create --config config_nni5.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081

# boruta for feature derivation
best_expid=BD3oGFia
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml 

# boruta for feature selection
best_expid=BD3oGFia
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml --best_db_path $best_db_path --features_for_derivation boruta_explog/09647097-60b1-4c47-bc04-47eb678f73ea/confirmed_vars.txt

