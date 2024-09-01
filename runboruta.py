
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import seaborn as sns
from utils import preprocess_data, load_data, load_config, augment_samples, custom_eval_roc_auc_factory
from best_params import opendb, get_best_params

def main(filepath, target_column, log_dir, groupingparams, params):
    paramcopy = params.copy()
    scale_factor = params.pop('scale_factor') # 用于线性缩放目标变量
    log_transform = params.pop('log_transform') # 是否对目标变量进行对数变换
    custom_metric_key = params.pop('custom_metric')
    n_estimators = params.pop('num_boost_round')
    topn = params.pop('topn', None)
    early_stopping_rounds = params.pop('early_stopping_rounds')
    data = load_data(filepath)
    params['reg_alpha'] = params.pop('alpha')
    params["device"] = "cuda"
    params["tree_method"] = "hist"

    X, y, sample_weight = preprocess_data(data, target_column, 
                                          scale_factor,log_transform, 
                                            groupingparams,
                                            pick_key= 'all',
                                           feature_derivation = None, 
                                           topn=None, sorted_features=None)
    
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(X, y, sample_weight, test_size=0.2, random_state=42)
    
    print(f"Before augmentation: {X_train.shape}, {y_train.shape}")
    X_train, y_train = augment_samples(X_train, y_train, sw_train)
    print(f"After augmentation: {X_train.shape}, {y_train.shape}")
    
    rf = xgb.XGBRegressor(n_estimators = n_estimators,**params)
                          
    # 初始化存储特征排名的 DataFrame
    ranking_df = pd.DataFrame(columns=X_train.columns)

    # 运行 Boruta 20 次
    for i in range(10):
        print(f"Iteration {i+1}")
        
        # 初始化Boruta特征选择器
        boruta_selector = BorutaPy(rf, n_estimators=n_estimators, verbose=2, 
                                   random_state=i, max_iter=25)
        
        # 对训练数据进行特征选择
        boruta_selector.fit(X_train, y_train)
        
        # 获取特征排名
        feature_ranks = boruta_selector.ranking_
        
        # 将特征排名保存到 DataFrame 中
        ranking_df.loc[i+1] = feature_ranks
    return ranking_df

def plot_boruta(ranking_df):
    numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

    median_values = numeric_ranking_df.median()
    sorted_columns = median_values.sort_values().index

    # 设置绘图风格
    plt.figure(figsize=(15, 8))
    sns.set(style="whitegrid")

    # 绘制箱线图
    sns.boxplot(data=numeric_ranking_df[sorted_columns], palette="Greens")

    plt.xticks(rotation=90)
    plt.title("Sorted Feature Ranking Distribution by Boruta", fontsize=16)
    plt.xlabel("Attributes", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'boruta.png'))
    plt.close()

if __name__ == "__main__":
    filepath = 'output/dataforxgboost.csv'
    target_column = 'VisitDuration'
    log_dir = 'boruta_explog'
    groupingparams = load_config('groupingsetting.yml')['groupingparams']
    best_db_path = 'nni1_explog/ZpoUyrIC/db/nni.sqlite'
    df  = opendb(best_db_path)
    ls_of_params = get_best_params(df, [('default','minimize')], 1)
    best_param = ls_of_params[0][1]

    ranking_df = main(filepath, target_column, log_dir, groupingparams, best_param)
    ranking_df.to_csv(os.path.join(log_dir, 'ranking_df.csv'))
    plot_boruta(ranking_df, log_dir)