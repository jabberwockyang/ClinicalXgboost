# todo  topn
import nni
import xgboost as xgb
from sklearn.model_selection import train_test_split
import argparse
import yaml
import os
from typing import Dict, List
import json
   
from best_params import opendb, get_best_params
from utils import preprocess_data, load_data, custom_eval_roc_auc_factory, save_checkpoint, evaluate_model, plot_feature_importance, convert_floats, LoadFeatures, sorted_features_list


# 主函数
def nnimain(filepath, target_column, log_dir, 
            groupingparams: Dict[str, List[str]] , 
            features_for_deri = None, sorted_features = None):
    # 从 NNI 获取超参数 
    params = nni.get_next_parameter()

    # Add GPU parameter
    params["device"] = "cuda"
    params["tree_method"] = "hist"

    scale_factor = params.pop('scale_factor') # 用于线性缩放目标变量
    log_transform = params.pop('log_transform') # 是否对目标变量进行对数变换
    custom_metric_key = params.pop('custom_metric')
    num_boost_round = params.pop('num_boost_round')
    early_stopping_rounds = params.pop('early_stopping_rounds')
    #pop topn parameter if no such key set to None
    topn = params.pop('topn', None)

    data = load_data(filepath)
    X, y, sample_weight = preprocess_data(data, target_column, 
                                          scale_factor,log_transform, 
                                            groupingparams,
                                            pick_key= 'all',
                                           feature_derivation = features_for_deri, 
                                           topn=topn, sorted_features=sorted_features)

    # 划分训练集 验证集 测试集
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weight, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(X_train, y_train, sw_train, test_size=0.2, random_state=42)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sw_train)
    dval = xgb.DMatrix(X_val, label=y_val, weight=sw_val)
    dtest = xgb.DMatrix(X_test, label=y_test, weight=sw_test)

    custom_metric = custom_eval_roc_auc_factory(custom_metric_key, scale_factor, log_transform) # 'prerec_auc' 'roc_auc' None

    # 训练模型
    model = xgb.train(params, 
                      dtrain, 
                      custom_metric = custom_metric,
                      evals = [(dtrain, 'train'), 
                               (dval, 'validation')],
                      maximize= True,
                      num_boost_round = num_boost_round,
                      early_stopping_rounds=early_stopping_rounds)

    # 获取当前的超参数配置ID 获取当前的实验ID
    sequence_id = nni.get_sequence_id()
    experiment_id = nni.get_experiment_id()
    
    # 保存模型checkpoint
    checkpoint_path = f'{log_dir}/{experiment_id}/checkpoint/{sequence_id}_model_checkpoint.json'
    save_checkpoint(model, checkpoint_path)

    # 保存模型结果
    result_dir = f'{log_dir}/{experiment_id}/result/{sequence_id}'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    loss, max_roc_auc, max_prerec_auc = evaluate_model(model, dtest, result_dir, scale_factor, log_transform)
    # 保存实验 id 超参数 和 结果 # 逐行写入
    paramandreuslt = params.copy()
    paramandreuslt['loss'] = loss
    paramandreuslt['max_roc_auc'] = max_roc_auc
    paramandreuslt['max_prerec_auc'] = max_prerec_auc
    paramandreuslt['sequence_id'] = sequence_id
    with open(f'{log_dir}/{experiment_id}/paramandresult.jsonl', 'a') as f:
        json.dump(convert_floats(paramandreuslt), f, ensure_ascii=False)
        f.write('\n')

    # 评估特征重要性
    plot_feature_importance(model, result_dir)
    # 向 NNI 报告结果
    nni.report_final_result({
        'default': loss,
        'roc_auc': max_roc_auc,
        'prerec_auc': max_prerec_auc
    })


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='clinical_data.csv', help='Path to the clinical data file')
    parser.add_argument('--target_column', type=str, default='disease_duration', help='Name of the target column')
    parser.add_argument('--exp_dir', type=str, default='explog', help='Path to the experiment log')
    parser.add_argument('--features_for_derivation', type=str, default=None, help='the path to the feature derivation file')
    parser.add_argument('--groupingparams', type=str, default=None, help='the path to the grouping parameters file')
    parser.add_argument('--importance_sorting', type=str, default=None, help='the path to the feature importance sorting file')
    args = parser.parse_args()
    return args

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    
    args = argparser()
    filepath = args.filepath
    target_column = args.target_column
    log_dir = args.exp_dir
    groupingparams = load_config(args.groupingparams)['groupingparams']
    # load features to use
    features_for_deri = LoadFeatures(args.features_for_derivation)
    # importance sorting of features and derivation features
    sorted_features = sorted_features_list(args.importance_sorting)

    nnimain(filepath, target_column, log_dir, groupingparams, features_for_deri, sorted_features)
