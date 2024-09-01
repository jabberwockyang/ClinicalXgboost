import warnings
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc, precision_recall_curve
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import json
from loguru import logger
import itertools 
import seaborn as sns
import re

# 数据加载
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def convert_floats(o):
    if isinstance(o, np.float32):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()  # Convert ndarray to list
    elif isinstance(o, dict):
        return {k: convert_floats(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [convert_floats(v) for v in o]
    return o

def LoadFeatures(filepath):
    if filepath is None:
        return None
    with open(filepath, 'r') as f:
        importancedata = json.load(f)
    alldata = [item for item in importancedata if item['label'] == 'all'][0]
    return alldata['top_n']

def sorted_features_list(importance_sorting_filepath):
    if importance_sorting_filepath is None:
        return None
    importance_df = pd.read_csv(importance_sorting_filepath)
    # group by feature and average weight
    # rename Feature to feature and Importance to weight
    importance_df = importance_df.rename(columns={'Feature': 'feature'})
    importance_df = importance_df.rename(columns={'Importance': 'weight'})
    # get feature and weight
    importance_df = importance_df[['feature', 'weight']]   
    # group by feature and average weight
    importance_df = importance_df.groupby('feature').mean().reset_index()
    # sort by weight from large to small
    importance_df = importance_df.sort_values(by='weight', ascending=False)
    return importance_df['feature'].tolist()
    

# 数据预处理
def preprocess_data(df, target_column, 
                    scale_factor,log_transform, 
                    groupingparams: Dict[str, List[str]],
                    pick_key = '0-2', 
                    feature_derivation = None,
                    topn = None, 
                    sorted_features = None):
    ## filtering data
    # remove rows with missing values in all result columns
    with open ('ExaminationItemClass_ID.json', 'r') as json_file:
        ExaminationItemClass = json.load(json_file)
    namelist = [x[1] for x in ExaminationItemClass['CommonBloodTest']]
    result_cols2 = df.columns[df.columns.str.contains("|".join(namelist))]
    df = df.dropna(subset=result_cols2, how='all') 
    # remove rows with missing values in target column
    df = df.dropna(subset=[target_column])
    # remove columns in results_col with only 1 unique value
    result_cols = df.columns[df.columns.str.contains('Avg|Count|Sum|Max|Min|Median|Std|Skew|Kurt|Pct')]
    df = df.drop(columns=result_cols[df[result_cols].nunique() == 1])

    # Group data by visitdurationgroup and calculate weights
    df['visitdurationgroup'] = pd.cut(df['VisitDuration'], 
                                      bins=[0, 42, 365, 1095, 1825,10000], 
                                      labels=["<6w", "6w-1y", "1-3y", "3-5y", "5y+"], ordered=True)
    df = df.dropna(subset=['visitdurationgroup'])
    weights = df['visitdurationgroup'].value_counts(normalize=True)
    df['sample_weight'] = df['visitdurationgroup'].map(lambda x: 1 / (weights[x] + 1e-10))

    # group data by agegroup
    bins = groupingparams['bins']
    labels = groupingparams['labels']

    df = df.assign(agegroup=pd.cut(df['FirstVisitAge'], bins= bins, labels= labels))
    
    df['agegroup'] = pd.Categorical(df['agegroup'], categories= labels, ordered=True)

    # Missing value imputation and other preprocessing
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    result_cols = df.columns[df.columns.str.contains('Avg|Count|Sum|Max|Min|Median|Std|Skew|Kurt|Pct')]
    overall_median = df[result_cols].median()
    df[result_cols] = df.groupby('agegroup', observed=True)[result_cols].transform(
        lambda x: x.fillna(x.median() if not np.isnan(x.median()) else overall_median)
    )
    warnings.filterwarnings("default", category=RuntimeWarning, message="Mean of empty slice")

    # subsetting data
    if pick_key != 'all':
        df = df[df['agegroup'] == pick_key]
    else:
        pass
    # scale the data by min-max 
    for col in result_cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Feature derivation
    if isinstance(feature_derivation, list):
        combinations = list(itertools.combinations(feature_derivation, 2))
        # 对于每一对特征组合，生成派生变量并添加到df中
        for (feat1, feat2) in combinations:
            # 创建新的派生变量列名
            new_col_name = f'{feat1}_div_{feat2}'
            df[new_col_name] = df[feat1] / (df[feat2] + 1e-10)

        print(f"Number of derived features: {len(combinations)}")
    
    # feature filtering by importance sorting
    features_to_use = [feat for feat in df.columns if feat not in [target_column, 'sample_weight','agegroup', 'visitdurationgroup']]
    if topn > 1: # topn is an integer
        topn = topn
    elif topn < 0: # topn is -1 use all features
        topn = len(features_to_use)
    else: # topn is a ratio between 0 and 1
        topn = round(len(features_to_use) * topn) 
    # rank features to use by sorted_features if not in features_to_use then place at the end
    features_to_use = [feat for feat in sorted_features if feat in features_to_use]  + [feat for feat in features_to_use if feat not in sorted_features]
    features_to_use = features_to_use[:topn]
    print(f"Number of features used: {len(features_to_use)}")
    print(f"Features used: {features_to_use[:5]} ...")


    print(df.head())
    # Generate data for training
    X = df[features_to_use]
    y = df[target_column]
    # scaled by scale_factor linearly
    y = np.round(y / scale_factor) + 1
    # log transform
    if log_transform == "log2":
        y = np.log2(y)
    elif log_transform == "log10":
        y = np.log10(y)
    else:
        y = y
    # get sample_weight
    sample_weight = df['sample_weight']
    logger.info(f"Data for 'agegroup' {pick_key} is ready, X shape: {X.shape}, y shape: {y.shape}")
    return X, y, sample_weight


def plot_roc_curve(fpr, tpr, auc, savepath):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'roc curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(savepath)

def plot_prc_curve(precision, recall, auc, savepath):
    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw, label=f'prc curve (area = %0.2f)' % auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve example')
    plt.legend(loc="lower right")
    plt.savefig(savepath)

def plot_feature_importance(model, savepath):
    dflist = [] 
    for target in ['weight', 'total_gain', 'total_cover', 'gain', 'cover']:
        importance = model.get_score(importance_type= target)
        importance_df = pd.DataFrame(index=importance.keys(), data={target: list(importance.values())})

        dflist.append(importance_df)
    importance_df = pd.concat(dflist, axis=1)
    importance_df['feature'] = importance_df.index  
    importance_df.to_csv(os.path.join(savepath, 'feature_importance.csv'), index=False)
    if importance_df.empty:
        print("No feature importance data available to plot.")
        return
    importance_df = importance_df.sort_values('weight', ascending=False)
    #scale to 0-1
    for col in ['weight', 'total_gain', 'total_cover', 'gain', 'cover']:
        importance_df[col] = importance_df[col] / importance_df[col].max()
    # plot first 25 features
    importance_df = importance_df.head(25)
    importance_df = importance_df.loc[:, ['feature', 'weight', 'total_gain', 'total_cover']]
    plt.figure()
    fig, ax = plt.subplots(figsize=(20, 10)) # figsize=(width, height) in inches
    ax.set_position([0.1, 0.3, 0.8, 0.6])  # [left, bottom, width, height] in figure coordinates

    importance_df.plot(kind='bar', ax = ax)
    plt.xlabel('Feature')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(os.path.join(savepath, 'feature_importance.png'))


# 自定义评估函数1
def prerec_auc_metric(y_test, y_pred):
    auc_list = []
    for binary_threshold in [42]:
        y_test_binary = np.where(y_test.copy() > binary_threshold, 1, 0)
        prec, rec, thresholds = precision_recall_curve(y_test_binary, y_pred) 
        prerec_auc = auc(rec, prec)
        auc_list.append(
            {"binary_threshold": binary_threshold,
            "precision": prec,
            "recall": rec,
            "thresholds": thresholds,
            "prerec_auc": prerec_auc})
    return auc_list

# 自定义评估函数2
def roc_auc_metric(y_test, y_pred):
    auc_list = []
    for binary_threshold in [42]:
        y_test_binary = np.where(y_test.copy() > binary_threshold, 1, 0)
        fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred)
        roc_auc = auc(fpr, tpr)
        auc_list.append(
            {"binary_threshold": binary_threshold,
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "roc_auc": roc_auc})
    return auc_list


# 模型评估
def evaluate_model(model, dtest, result_dir,scale_factor, log_transform):
    y_test = dtest.get_label()
    y_pred = model.predict(dtest)

    if log_transform == "log2":
        y_test_reversed = 2 ** y_test
        y_pred_reversed = 2 ** y_pred
    elif log_transform == "log10":
        y_test_reversed = 10 ** y_test
        y_pred_reversed = 10 ** y_pred
    else:
        y_test_reversed = y_test
        y_pred_reversed = y_pred
    
    # reverse scale_factor
    y_test_reversed = (y_test_reversed - 1) * scale_factor
    y_pred_reversed = (y_pred_reversed - 1) * scale_factor

    # loss
    loss = mean_squared_error(y_test_reversed, y_pred_reversed)

    # roc auc
    roc_auc_json = roc_auc_metric(y_test_reversed, y_pred_reversed)
    max_roc_auc = max([roc_obj["roc_auc"] for roc_obj in roc_auc_json])

    # prerec auc
    prerec_auc_json = prerec_auc_metric(y_test_reversed, y_pred_reversed)
    max_prerec_auc = max([prerec_obj["prerec_auc"] for prerec_obj in prerec_auc_json])

    # save result
    auc_results = {
        "roc_auc": roc_auc_json,
        "prerec_auc": prerec_auc_json
    }
    with open(os.path.join(result_dir, 'auc_results.json'), 'w') as f:
        json.dump(convert_floats(auc_results), f, ensure_ascii=False, indent=4)

    # save plot
    for item in roc_auc_json:
        binary_threshold = item["binary_threshold"]
        fpr = item["fpr"]
        tpr = item["tpr"]
        roc_auc = item["roc_auc"]

        save_path = os.path.join(result_dir, f'{binary_threshold}_roc.png')
        plot_roc_curve(fpr, tpr, roc_auc, savepath=save_path)

    for item in prerec_auc_json:
        binary_threshold = item["binary_threshold"]
        prec = item["precision"]
        rec = item["recall"]
        prerec_auc = item["prerec_auc"]

        save_path = os.path.join(result_dir, f'{binary_threshold}_prc.png')
        plot_prc_curve(rec, prec, prerec_auc, savepath=save_path)

    return loss, max_roc_auc, max_prerec_auc

# 保存模型checkpoint
def save_checkpoint(model, checkpoint_path):
    # Create directory if not exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Save
    model.save_model(checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def custom_eval_roc_auc_factory(custom_metric_key, scale_factor, log_transform):
    if custom_metric_key == 'prerec_auc':
        # 用于传入 train_model 的自定义评估函数1
        def custom_eval_prerec_auc(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            '''
            In the above code snippet, squared_log is the objective function we want. 
            It accepts a numpy array predt as model prediction, and the training DMatrix for obtaining required information, including labels and weights (not used here). 
            This objective is then used as a callback function for XGBoost during training by passing it as an argument to xgb.train:
            '''
            y_train = dtrain.get_label()
            y_pred = predt
            if log_transform == "log2":
                y_train_reversed = 2 ** y_train
                y_pred_reversed = 2 ** y_pred
            elif log_transform == "log10":
                y_train_reversed = 10 ** y_train
                y_pred_reversed = 10 ** y_pred
            else:
                y_train_reversed = y_train
                y_pred_reversed = y_pred
            # reverse scale_factor
            y_train_reversed = (y_train_reversed - 1) * scale_factor

            auc_json = prerec_auc_metric(y_train_reversed, y_pred_reversed)
            max_prerec_auc = max([auc_obj['prerec_auc'] for auc_obj in auc_json])
            return 'prerec_auc', max_prerec_auc
        return custom_eval_prerec_auc
    elif custom_metric_key == 'roc_auc':
        # 用于传入 train_model 的自定义评估函数2
        def custom_eval_roc_auc(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
            '''
            In the above code snippet, squared_log is the objective function we want. 
            It accepts a numpy array predt as model prediction, and the training DMatrix for obtaining required information, including labels and weights (not used here). 
            This objective is then used as a callback function for XGBoost during training by passing it as an argument to xgb.train:
            '''
            y_train = dtrain.get_label()
            y_pred = predt
            if log_transform == "log2":
                y_train_reversed = 2 ** y_train
                y_pred_reversed = 2 ** y_pred
            elif log_transform == "log10":
                y_train_reversed = 10 ** y_train
                y_pred_reversed = 10 ** y_pred
            else:
                y_train_reversed = y_train
                y_pred_reversed = y_pred
            # reverse scale_factor
            y_train_reversed = (y_train_reversed - 1) * scale_factor
            auc_json = roc_auc_metric(y_train_reversed, y_pred_reversed)
            max_roc_auc = max(auc_obj['roc_auc'] for auc_obj in auc_json)
            return 'roc_auc', max_roc_auc
        return custom_eval_roc_auc
    
    elif  custom_metric_key == None:
        return None

def parse_gr_results(gr_results):
    '''
    input: jsonfile path
    output: dataframe with max_roc_auc and group

    '''
    with open(gr_results, 'r') as f:
        results = json.load(f)
    alllist = []
    paramresults = [r['paramresult'] for r in results]
    for paramresult in paramresults:
        alllist.extend(paramresult)
    df = pd.DataFrame(alllist)
    df= df[['group', 'max_roc_auc']]
    return df

def parse_nni_results(nni_results, metric:str, minimize:bool, number_of_trials:int):
    '''
    input: jsonfile path
    output: dataframe with max_roc_auc and group

    '''
    if metric == 'default':
        metric = 'loss'
    with open(nni_results, 'r') as f:
        results = [json.loads(line) for line in f.readlines()]
    df = pd.DataFrame({
    'max_roc_auc': [r['max_roc_auc'] for r in results],
    'metric': [r[metric] for r in results],
    'group': ['all' for r in results],
    })
    df = df.sort_values(by='metric', ascending=minimize)
    df = df.head(number_of_trials)
    df.drop(columns=['metric'], inplace=True)
    return df

def plot_roc_summary(df, outdir):

    # plot dot plot max_roc_auc in different group x axis is group y axis is max_roc_auc
    # different objective with different color
    df['max_roc_auc'] = df['max_roc_auc'].astype(float)
    uniquegroups = df['group'].unique() 
    orderedlist = sorted(uniquegroups, key = lambda x: int(re.split(r'[-+]', x)[0] if re.match(r'^\d', x) else 9999))
    df['group'] = pd.Categorical(df['group'], categories = orderedlist, ordered = True)

    plt.figure(figsize=(6, 5))
    # violinplot with dots  
    sns.violinplot(data = df, x = 'group', y = 'max_roc_auc')
    sns.stripplot(data = df, x = 'group', y = 'max_roc_auc', color = 'orange', size = 6, jitter = 0.25)
    plt.xlabel('group')
    plt.ylabel('max_roc_auc')
    plt.title('max_roc_auc in different group')
    plt.savefig(os.path.join(outdir, 'max_roc_auc.png'))
    plt.close()
