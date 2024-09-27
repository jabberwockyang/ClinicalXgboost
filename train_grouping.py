# todo 全部数据用作validation
# 全部数据用作画图结果
import xgboost as xgb
from sklearn.model_selection import train_test_split
import argparse
import yaml
import os
from typing import Dict, List
import json
from loguru import logger
   
from best_params import opendb, get_best_params
from utils import load_data, custom_eval_roc_auc_factory, save_checkpoint, evaluate_model, plot_feature_importance, convert_floats, load_feature_list_from_boruta_file
from preprocessor import Preprocessor, FeatureDrivator, FeatureFilter
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import numpy as np

# 主函数
def main(filepath, preprocessor, log_dir, params, label_toTrain: List[str], n_splits=5):
    paramcopy = params.copy()
    model_type = params.pop('model')
    
    scale_factor = params.pop('scale_factor') # 用于线性缩放目标变量
    log_transform = params.pop('log_transform') # 是否对目标变量进行对数变换
    topn = params.pop('topn', None)
    data = load_data(filepath)

    for k in label_toTrain:
        if not os.path.exists(f'{log_dir}/{k}'):
            os.makedirs(f'{log_dir}/{k}')
        X, y, sample_weight = preprocessor.preprocess(data,
                                                    scale_factor,
                                                    log_transform,
                                                    pick_key= k,
                                                    topn=topn) # topn in transmitted anyway when no topn in searchspace is set to None
        
        # concat x and y save to the log_dir
        Xy = X.copy()
        Xy['target'] = y
        Xy.to_csv(f'{log_dir}/{k}/datapreprocessed.csv', index=False)
        logger.info(f"Data preprocessed for {k} and saved")

        if X.shape[0] == 0:
            logger.info(f"No data for {k}")
            continue
        # 划分训练集 验证集 测试集
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weight, test_size=0.2, random_state=42)
        # 初始化k折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        fold_ext = 1

        # k fold cross validation internal test set
        for train_index, val_index in kf.split(X_train):

            result_dir = f'{log_dir}/{k}/result/{fold_ext}'
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            X_train_int, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_int, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            sw_train_int, sw_val = sw_train.iloc[train_index], sw_train.iloc[val_index]
            
            def train_model(model_type, params):
                if model_type == "xgboost":
                    # XGBoost 特有的 GPU 参数
                    params["device"] = "cuda"
                    params["tree_method"] = "hist"

                    custom_metric_key = params.pop('custom_metric')
                    num_boost_round = params.pop('num_boost_round')
                    early_stopping_rounds = params.pop('early_stopping_rounds')
                    custom_metric, maximize = custom_eval_roc_auc_factory(custom_metric_key, scale_factor, log_transform) # 'prerec_auc' 'roc_auc' None

                    xgb_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                    # 构建 DMatrix
                    dtrain = xgb.DMatrix(X_train_int, label=y_train_int, weight=sw_train_int)
                    dval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)

                    # 训练 XGBoost 模型，传入早停参数
                    model = xgb.train(xgb_params, 
                                    dtrain, 
                                    custom_metric=custom_metric,
                                    evals=[(dtrain, 'train'), 
                                        (dval, 'validation')],
                                    maximize=maximize,
                                    num_boost_round=num_boost_round,
                                    early_stopping_rounds=early_stopping_rounds)
                    logger.info(f"Model trained for {k}")
                
                elif model_type == "svm":
                    svm_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                    model = svm.SVR(**svm_params)
                    model.fit(X_train, y_train, sample_weight=sw_train)

                elif model_type == "random_forest":
                    rf_params = {k: v for k, v in params.items() if v is not None}  # 去除 None
                    model = RandomForestRegressor(**rf_params)
                    model.fit(X_train, y_train, sample_weight=sw_train)

                else:
                    raise ValueError(f"Unsupported model type: {model_type}")
                
                return model

            result_dir = f'{log_dir}/{k}/result/{fold}'
            final_marker = os.path.join(result_dir, 'feature_importance.png')
            if os.path.exists(final_marker):
                logger.info(f"Model already trained for {k} fold {fold}")
                continue

            model = train_model(model_type, params.copy())  # 传入超参数

            # 保存模型checkpoint
            checkpoint_path = f'{log_dir}/{k}/checkpoint/{fold}_model_checkpoint.json'
            save_checkpoint(model, checkpoint_path)
            logger.info(f"Model checkpoint saved for {k}")

            # external 5 fold cross validation test set
            kf_ext = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_results = []
            fold_int = 1
        
            for train_index, test_index in kf_ext.split(X):
                X_train_ext, X_test_ext = X.iloc[train_index], X.iloc[test_index]
                y_train_ext, y_test_ext = y.iloc[train_index], y.iloc[test_index]
                sw_train_ext, sw_test_ext = sample_weight[train_index], sample_weight[test_index]
                dtest = xgb.DMatrix(X_train_ext, label=y_train_ext, weight=sw_train_ext)

                # 保存模型结果
                loss, max_roc_auc, max_prerec_auc = evaluate_model(model, dtest, result_dir, scale_factor, log_transform)
                fold_results.append({
                    'fold': fold_int,
                    'loss': loss,
                    'max_roc_auc': max_roc_auc,
                    'max_prerec_auc': max_prerec_auc
                })
                fold_int += 1
            
            avg_loss = np.mean([result['loss'] for result in fold_results])
            avg_roc_auc = np.mean([result['max_roc_auc'] for result in fold_results])
            avg_prerec_auc = np.mean([result['max_prerec_auc'] for result in fold_results])

            # 保存实验 id 超参数 和 结果 # 逐行写入
            paramandreuslt = paramcopy.copy()
            paramandreuslt['group'] = k
            paramandreuslt['loss'] = avg_loss
            paramandreuslt['loss_list'] = [result['loss'] for result in fold_results]

            paramandreuslt['max_roc_auc'] = avg_roc_auc
            paramandreuslt['max_roc_auc_list'] = [result['max_roc_auc'] for result in fold_results]

            paramandreuslt['max_prerec_auc'] = avg_prerec_auc
            paramandreuslt['max_prerec_auc_list'] = [result['max_prerec_auc'] for result in fold_results]

            paramandreuslt['fold'] = fold_ext
            with open(f'{log_dir}/{k}/paramandresult.json', 'w') as f:
                json.dump(convert_floats(paramandreuslt), f, ensure_ascii=False, indent=4)
    
            # 评估特征重要性
            plot_feature_importance(model, result_dir)
            
            logger.info(f"Model result saved for {k} fold {fold_ext}")
            fold_ext+=1
        logger.info(f"Model result saved for {k}")

def parse_args():
    parser = argparse.ArgumentParser(description='Run XGBoost Model Training with Grouping Parameters')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    parser.add_argument('--expid', type=str, default=None, help='Experiment ID of nni results to run grouped training')
    
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        nni_config = config['nni']
        train_config = config['train']
    return config, nni_config, train_config

if __name__ == "__main__":
    args = parse_args()
    config, nni_config, train_config = load_config(args.config)

    # 读取nni实验中最佳超参数
    former_exp_stp = nni_config['exp_stp']
    # if expid not provided in arg then use the one in config
    best_exp_id = args.expid if args.expid else nni_config['best_exp_id']
    metric_to_optimize = nni_config['metric_to_optimize']
    number_of_trials = nni_config['number_of_trials']

    best_db_path = f'{former_exp_stp}/{best_exp_id}/db/nni.sqlite'
    df = opendb(best_db_path)
    ls_of_params = get_best_params(df, metric_to_optimize, number_of_trials)
    

    # 训练数据相关参数
    current_exp_stp = train_config['exp_stp']
    if not os.path.exists(current_exp_stp):
        os.makedirs(current_exp_stp)
    filepath = train_config['filepath']
    target_column = train_config['target_column']
    
    # 分组参数
    label_toTrain = train_config['label_toTrain']
    groupingparams = {'grouping_parameter_id': train_config['grouping_parameter_id'],
                      'bins': train_config['groupingparams']['bins'],
                      'labels': train_config['groupingparams']['labels']}

    # 可选参数处理
    # feature derivation
    features_for_deri = load_feature_list_from_boruta_file(train_config['features_for_derivation']) if train_config['features_for_derivation'] else None
    # feature selection
    method =  train_config['variable_selection_method']
    f = train_config['features_list'].split("/")[1].split('-')[0] if method else None
    filterationparam = f'{method}_{f}' if f else None
    d = train_config['features_for_derivation'].split("/")[1].split("-")[0] if features_for_deri else None
    derivativeparam = f'deri_{d}' if d else None
    if method is None:
        pass
    elif method == 'sorting':
        # must have topn in search space
        logger.debug("Using sorting method for variable selection, please make sure search space contains topn parameter")
        # must provide features_list
        assert train_config['features_list'] is not None, "features_list must be provided for sorting method"
        sorted_features = load_feature_list_from_boruta_file(train_config['features_list'])
    elif method == 'selection':
        logger.debug("Using selection method for variable selection, please make sure search space contains features_list parameter")
        assert train_config['features_list'] is not None, "features_list must be provided for selection method"
        sorted_features = load_feature_list_from_boruta_file(train_config['features_list'])
    else:
        raise ValueError(f"Invalid variable selection method: {method}, please choose from 'sorting' or 'selection'")
    # 实例化特征衍生和特征选择

    fd = FeatureDrivator(features_for_deri) if features_for_deri else None
    ff = FeatureFilter(target_column, method= method, features_list=sorted_features) if method else None
    # 实例化预处理器
    pp = Preprocessor(target_column, groupingparams,
                      feature_derive=fd,
                      FeaturFilter=ff)
    # 实验日志目录
    experiment_id = f'{best_exp_id}_{"&".join([m[0] for m in metric_to_optimize])}_top{number_of_trials}_gr{train_config["grouping_parameter_id"]}'
    experiment_id += f"_{filterationparam}" if filterationparam else ''
    experiment_id += f"_{derivativeparam}" if derivativeparam else ''
    if not os.path.exists(f'{current_exp_stp}/{experiment_id}'):
        os.makedirs(f'{current_exp_stp}/{experiment_id}')
    topparams = [p[1] for p in ls_of_params]
    with open(f'{current_exp_stp}/{experiment_id}/topparams.json', 'w') as f:
        json.dump(convert_floats(topparams), f, ensure_ascii=False, indent=4)
   
    for best_param_id, best_params, sequence_ids in ls_of_params:
        foldername = str(best_exp_id)+ '_' + str(best_param_id) + '_' + str(train_config['grouping_parameter_id']) 
        log_dir = f'{current_exp_stp}/{experiment_id}/{foldername}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 训练模型
        main(filepath, pp, log_dir, best_params, label_toTrain)



