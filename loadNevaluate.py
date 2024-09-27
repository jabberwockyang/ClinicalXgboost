import xgboost as xgb
from utils import evaluate_model, load_data
from best_params import opendb, get_best_params
from preprocessor import Preprocessor, FeatureDrivator, FeatureFilter
from sklearn.model_selection import KFold

import os


def load_checkpoint(log_dir, experiment_id, sequence_id):
    checkpoint = f'{log_dir}/{experiment_id}/checkpoint/{sequence_id}_model_checkpoint.json'
    model = xgb.XGBRegressor()
    model.load_model(checkpoint)
    return model

def main(filepath, log_dir, experiment_id, metric_to_optimize, number_of_trials, preprocessor):
    # get best params
    best_db_path = os.path.join(log_dir,experiment_id, 'db','nni.sqlite')
    df = opendb(best_db_path)
    ls_of_params = get_best_params(df, metric_to_optimize, number_of_trials)

    # load test data
    data = load_data(filepath)


    for param_id, params, sequence_ids in ls_of_params:     # [(param_id, params, sequence_ids)]
        model = load_checkpoint(log_dir,experiment_id, sequence_ids[0])
        model_type = params.pop('model_type')
        scale_factor = params.pop('scale_factor')
        log_transform = params.pop('log_transform')
        topn = params.pop('topn', None)

        X, y, sample_weight = preprocessor.preprocess(data,
                                                    scale_factor,
                                                    log_transform,
                                                    pick_key= "all",
                                                    topn=topn) 

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        fold = 1
        for train_index, val_index in kf.split(X):
            X_test, X_val = X.iloc[train_index], X.iloc[val_index]
            y_test, y_val = y.iloc[train_index], y.iloc[val_index]
            sw_test, sw_val = sample_weight.iloc[train_index], sample_weight.iloc[val_index]

            loss, max_roc_auc, max_prerec_auc = evaluate_model(model, model_type, X_test, y_test, sw_test,
                                                                scale_factor, log_transform)
            
            

if __name__ == '__main__':
        pp = Preprocessor(target_column, groupingparams,
                      feature_derive=fd,
                      FeaturFilter=ff)