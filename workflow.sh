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



# nni with derived variables nni2_explog
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





# nni with derived variables and topn nni3_explog
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




# nni with topn nni4_explog
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

#during boruta discover problem with imputation run nni1 again and grouping and importance not done

# BORUTA shows good performance in feature selection   

# a new workflow
# feature engingeering with minmxavg and acute chronic avg
###  done in sqlquery

# nni5 GET best params for boruta
nnictl create --config config_nni5.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081
# run importance for nni5
for expid in nni5_explog/*; do python3 importance.py --nnidir $expid --metric default --minimize True --number_of_trials 7; done


# boruta selection with no derived features
best_expid=BD3oGFia
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml 

# boruta selection with derived features
best_expid=BD3oGFia
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml --best_db_path $best_db_path --features_for_derivation boruta_explog/09647097-60b1-4c47-bc04-47eb678f73ea/confirmed_vars.txt

# found that boruta select less features with derived features
2024-09-13 14:06:23.031 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 42 common variables in all the lists
boruta selection with no derived features
['PlateletCount', 'MeanCorpuscularHemoglobinConcentration', 'AntiJo1', 'AbsoluteEosinophilCount', 'AntiScl70', 'CReactiveProtein', 'BasophilsPercentage', 'Histone', 'Plateletcrit', 'LymphocytesPercentage', 'TotalThyroxine', 'MeanPlateletVolume', 'EosinophilsPercentage', 'MeanCorpuscularHemoglobin', 'AntiDoubleStrandedDNA', 'DogDander', 'AntiSSA', 'WhiteBloodCellCount', 'ProliferatingCellNuclearAntigen', 'SMRNP', 'EggWhite', 'RedCellDistributionWidth', 'TotalTriiodothyronine', 'AbsoluteNeutrophilCount', 'ThyroidStimulatingHormone', 'Ragweed', 'MeanCorpuscularVolume', 'NeutrophilsPercentage', 'RedCellDistributionWidthCV', 'Hemoglobin', 'MonocytesPercentage', 'AbsoluteLymphocyteCount', 'AbsoluteMonocyteCount', 'AntiM2', 'AbsoluteBasophilCount', 'FreeThyroxine', 'AntiSM', 'EosinophilCountAbsolute', 'Cockroach', 'ImmunoglobulinE', 'Ro52', 'PlateletDistributionWidth']
2024-09-13 14:06:23.036 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 26 common variables in all the lists
boruta selection with derived features
['PlateletCount', 'MeanCorpuscularHemoglobinConcentration', 'AbsoluteEosinophilCount', 'AntiScl70', 'BasophilsPercentage', 'LymphocytesPercentage', 'TotalThyroxine', 'MeanPlateletVolume', 'EosinophilsPercentage', 'MeanCorpuscularHemoglobin', 'DogDander', 'ProliferatingCellNuclearAntigen', 'TotalTriiodothyronine', 'AbsoluteNeutrophilCount', 'Ragweed', 'MeanCorpuscularVolume', 'NeutrophilsPercentage', 'Hemoglobin', 'MonocytesPercentage', 'AbsoluteLymphocyteCount', 'AbsoluteMonocyteCount', 'AntiM2', 'AbsoluteBasophilCount', 'EosinophilCountAbsolute', 'ImmunoglobulinE', 'PlateletDistributionWidth']
2024-09-13 14:06:23.039 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 19 common variables in all the lists
boruta selection with derived features try2
['PlateletCount', 'MeanCorpuscularHemoglobinConcentration', 'AbsoluteEosinophilCount', 'AntiScl70', 'BasophilsPercentage', 'LymphocytesPercentage', 'MeanPlateletVolume', 'EosinophilsPercentage', 'MeanCorpuscularHemoglobin', 'DogDander', 'ProliferatingCellNuclearAntigen', 'AbsoluteNeutrophilCount', 'Ragweed', 'MeanCorpuscularVolume', 'MonocytesPercentage', 'AbsoluteLymphocyteCount', 'AntiM2', 'AbsoluteBasophilCount', 'PlateletDistributionWidth']

# retry boruta selection with derived features with larger number of iterations
best_expid=BD3oGFia
best_db_path=nni5_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_ac.csv --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml --best_db_path $best_db_path --features_for_derivation boruta_explog/09647097-60b1-4c47-bc04-47eb678f73ea/confirmed_vars.txt --max_iteration 100

2024-09-13 15:21:49.931 | INFO     | __main__:load_feature_list_from_boruta_file:490 - Found 30 common variables in all the lists
boruta selection with derived features try with maxiteration 100
['AbsoluteLymphocyteCount', 'ImmunoglobulinE', 'MeanPlateletVolume', 'AbsoluteNeutrophilCount', 'NeutrophilsPercentage', 'Hemoglobin', 'ThyroidStimulatingHormone', 'WhiteBloodCellCount', 'CReactiveProtein', 'MeanCorpuscularHemoglobinConcentration', 'AbsoluteBasophilCount', 'DogDander', 'Ragweed', 'Cockroach', 'EosinophilCountAbsolute', 'EosinophilsPercentage', 'LymphocytesPercentage', 'AntiM2', 'MonocytesPercentage', 'PlateletCount', 'Plateletcrit', 'PlateletDistributionWidth', 'MeanCorpuscularVolume', 'AntiScl70', 'MeanCorpuscularHemoglobin', 'AbsoluteMonocyteCount', 'ProliferatingCellNuclearAntigen', 'TotalTriiodothyronine', 'AbsoluteEosinophilCount', 'BasophilsPercentage']
# conclusion: no derived features were selected by boruta 

# group training with nni5 results
for yml in grouping_nni5_*.yaml; do
    for expid in nni5_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# group training with nni5 results + bselected features
for yml in grouping_nni5_bselected_*.yaml; do
    for expid in nni5_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni5_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done

# what is the clinical significance ?
# use acute data only

# nni6 use acute data only
nnictl create --config config_nni6.yml --port 8081 # ZpoUyrIC  nnictl resume ZpoUyrIC --port 8081

# boruta selection with acute data only
best_expid=
best_db_path=nni6_explog/$best_expid/db/nni.sqlite
python3 runboruta.py --filepath output/dataforxgboost_a.csv --best_db_path $best_db_path --target_column VisitDuration --log_dir boruta_explog --groupingparams groupingsetting.yml

# group training with nni6 results
for yml in grouping_nni6_*.yaml; do
    for expid in nni6_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# group training with nni6 results + bselected features
for yml in grouping_nni6_bselected_*.yaml; do
    for expid in nni6_explog/*; do
        base_expid=$(basename "$expid")
        python3 train_grouping.py --config "$yml" --expid "$base_expid"
    done
done

# Run importance.py for each grouping experiment folder
for expid in nni6_explog/*; do
    base_expid=$(basename "$expid")  # Extract base experiment ID
    for grfolder in gr_explog/${base_expid}*; do
        python3 importance.py --grdir "$grfolder"
    done
done