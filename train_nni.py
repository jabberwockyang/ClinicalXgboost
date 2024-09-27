# todo
import nni
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

from sklearn.model_selection import train_test_split
import argparse
import os
import json
from loguru import logger
from utils import load_data, custom_eval_roc_auc_factory, evaluate_model, convert_floats, load_config, load_feature_list_from_boruta_file
from preprocessor import Preprocessor, FeatureDrivator, FeatureFilter
from sklearn.model_selection import KFold
import numpy as np

# 主函数
def nnimain(filepath, log_dir, preprocessor: Preprocessor, n_splits=5):
    # 从 NNI 获取超参数 

    params = nni.get_next_parameter()
    paramandreuslt = params.copy() # 备份超参数
    # 从超参数中提取预处理参数
    scale_factor = params.pop('scale_factor') # 用于线性缩放目标变量
    log_transform = params.pop('log_transform') # 是否对目标变量进行对数变换
    topn = params.pop('topn', None) # pop topn parameter if no such key set to None # topn in search space can either be set as a list of int greater than 1 with -1 suggesting using all features or a float between 0 and 1 with 1 suggesting using all features 
    model_type = params.pop('model')

    data = load_data(filepath)
    X, y, sample_weight = preprocessor.preprocess(data, 
                                                  scale_factor,
                                                  log_transform,
                                                  pick_key= 'all',
                                                  topn=topn)
    
    # 备份数据
    Xy = X.copy()
    Xy['target'] = y
    experiment_id = nni.get_experiment_id()
    Xy.to_csv(f'{log_dir}/{experiment_id}/datapreprocessed.csv', index=False)

    # external test set
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weight, test_size=0.3, random_state=42)   
    
    # 初始化 KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    fold = 1

    # k fold cross validation internal test set
    for train_index, val_index in kf.split(X_train):
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

        model = train_model(model_type, params.copy())  # 传入超参数

        # 获取当前的超参数配置ID 获取当前的实验ID
        sequence_id = nni.get_sequence_id()
        experiment_id = nni.get_experiment_id()

        loss, max_roc_auc, max_prerec_auc = evaluate_model(model, model_type, X_test, y_test, sw_test,
                                                        scale_factor, log_transform)
        fold_results.append({
            'fold': fold,
            'loss': loss,
            'max_roc_auc': max_roc_auc,
            'max_prerec_auc': max_prerec_auc
        })
        fold += 1
    
    avg_loss = np.mean([result['loss'] for result in fold_results])
    avg_roc_auc = np.mean([result['max_roc_auc'] for result in fold_results])
    avg_prerec_auc = np.mean([result['max_prerec_auc'] for result in fold_results])

    
    # 保存实验 id 超参数 和 结果 # 逐行写入
    paramandreuslt['loss'] = avg_loss
    paramandreuslt['loss_list'] = [result['loss'] for result in fold_results]

    paramandreuslt['max_roc_auc'] = avg_roc_auc
    paramandreuslt['roc_auc_list'] = [result['max_roc_auc'] for result in fold_results]

    paramandreuslt['max_prerec_auc'] = avg_prerec_auc
    paramandreuslt['prerec_auc_list'] = [result['max_prerec_auc'] for result in fold_results]
    
    paramandreuslt['sequence_id'] = sequence_id

    with open(f'{log_dir}/{experiment_id}/paramandresult.jsonl', 'a') as f:
        json.dump(convert_floats(paramandreuslt), f, ensure_ascii=False)
        f.write('\n')

    # 向 NNI 报告结果
    nni.report_final_result({
        'default': avg_loss,
        'loss': avg_loss,
        'roc_auc': avg_roc_auc,
        'prerec_auc': avg_prerec_auc
    })


def argparser():
    parser = argparse.ArgumentParser()
    # 必填参数
    parser.add_argument('--filepath', type=str, help='Path to the clinical data file')
    parser.add_argument('--target_column', type=str, help='Name of the target column')
    parser.add_argument('--exp_dir', type=str, help='Path to the experiment log')
    parser.add_argument('--groupingparams', type=str, help='the path to the grouping parameters file')
    # 可选参数
    parser.add_argument('--features_for_derivation', type=str, default=None, help='the path to the feature derivation file')
    parser.add_argument('--variable_selection_method', type=str, default=None, help='Method for feature selection sorting or selection default None')
    parser.add_argument('--features_list', type=str, default=None, help='the path to the feature list file when using selection method it is used to load selected features when using sorting method it is used to load sorted features')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = argparser()
    # 必填参数
    filepath = args.filepath
    target_column = args.target_column
    log_dir = args.exp_dir
    groupingparams = load_config(args.groupingparams)['groupingparams']

    # 可选参数处理
    # feature derivation
    features_for_deri = load_feature_list_from_boruta_file(args.features_for_derivation) if args.features_for_derivation else None
    # feature selection
    method = args.variable_selection_method
    if method is None:
        pass
    elif method == 'sorting':
        # must have topn in search space
        logger.debug("Using sorting method for variable selection, please make sure search space contains topn parameter")
        # must provide features_list
        assert args.features_list is not None, "features_list must be provided for sorting method"
        sorted_features = load_feature_list_from_boruta_file(args.features_list)
    elif method == 'selection':
        logger.debug("Using selection method for variable selection, please make sure search space contains features_list parameter")
        assert args.features_list is not None, "features_list must be provided for selection method"
        sorted_features = load_feature_list_from_boruta_file(args.features_list)
    else:
        raise ValueError(f"Invalid variable selection method: {method}, please choose from 'sorting' or 'selection'")
    # 实例化特征衍生和特征选择
    fd = FeatureDrivator(features_for_deri) if features_for_deri else None
    ff = FeatureFilter(target_column, method= method, features_list=sorted_features) if method else None
    # 实例化预处理器
    pp = Preprocessor(target_column, groupingparams,
                      feature_derive=fd,
                      FeaturFilter=ff)
    # 运行主函数
    nnimain(filepath, log_dir, pp)
