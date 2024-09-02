
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import seaborn as sns
from utils import preprocess_data, load_data, load_config, augment_samples
from best_params import opendb, get_best_params
from preprocessor import Preprocessor
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import queue


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

    # X, y, sample_weight = preprocess_data(data, target_column, 
    #                                       scale_factor,log_transform, 
    #                                         groupingparams,
    #                                         pick_key= 'all',
    #                                        feature_derivation = None, 
    #                                        topn=None, sorted_features=None)
    
    preprocessor = Preprocessor(target_column, groupingparams)
    
    X, y, sample_weight = preprocessor.preprocess(data,
                                                  scale_factor,
                                                  log_transform,
                                                  pick_key= 'all')
    # 权重
    logger.info(f"Before augmentation: {X.shape}, {y.shape}")
    X, y = augment_samples(X, y, sample_weight)
    logger.info(f"After augmentation: {X.shape}, {y.shape}")

    # 初始化一个 Xgboost 回归模型
    rf = xgb.XGBRegressor(n_estimators = n_estimators,**params)
    # 初始化一个 DataFrame 来存储特征排名
    ranking_df = pd.DataFrame(columns=X.columns)
    output_queue = queue.Queue()

    def run_boruta(X, y, i):
        print(f"Iteration {i+1}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
        # 初始化Boruta特征选择器
        boruta_selector = BorutaPy(rf, n_estimators=n_estimators, verbose=2, 
                                   random_state=i, max_iter=25)
    
        boruta_selector.fit(X_train, y_train)
        feature_ranks = boruta_selector.ranking_
        output_queue.put((i+1, feature_ranks))

    # 使用多线程执行 Boruta
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_boruta, X, y, i) for i in range(20)]
        for future in futures:
            future.result()  # 等待所有线程完成

    # 从队列中获取结果并更新 ranking_df
    while not output_queue.empty():
        i, feature_ranks = output_queue.get()
        ranking_df.loc[i] = feature_ranks

    return ranking_df

def plot_boruta(ranking_df, log_dir, name = 'boruta'):
    numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')

    median_values = numeric_ranking_df.median()
    sorted_columns = median_values.sort_values().index

    # 设置绘图风格
    plt.figure(figsize=(25, 8))
    sns.set_theme(style="whitegrid")

    # 绘制箱线图
    sns.boxplot(data=numeric_ranking_df[sorted_columns], palette="Greens")
    # invert the y axis
    plt.gca().invert_yaxis()
    plt.xticks(rotation=90)
    plt.title("Sorted Feature Ranking Distribution by Boruta", fontsize=16)
    plt.xlabel("Attributes", fontsize=14)
    plt.ylabel("Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{name}.png'))
    plt.close()

def plot_boruta_by_group(ranking_df, log_dir):
    numeric_ranking_df = ranking_df.apply(pd.to_numeric, errors='coerce')
    features = numeric_ranking_df.columns
    groups = set([f.split('_')[0] for f in features])
    # make a new df to store the median ranking for each group
    group_ranking_df = pd.DataFrame(columns=list(groups))   

    for group in groups:
        # Filter features for the current group
        group_features = [f for f in features if f.startswith(group)]
        group_ranking_df[group] = numeric_ranking_df[group_features].max(axis=1)
    
    plot_boruta(group_ranking_df, log_dir, name='boruta_by_group')

if __name__ == "__main__":
    # filepath = 'output/dataforxgboost.csv'
    # target_column = 'VisitDuration'
    log_dir = 'boruta_explog'
    # groupingparams = load_config('groupingsetting.yml')['groupingparams']
    # best_db_path = 'nni1_explog/ZpoUyrIC/db/nni.sqlite'
    # df  = opendb(best_db_path)
    # ls_of_params = get_best_params(df, [('default','minimize')], 1)
    # best_param = ls_of_params[0][1]

    # ranking_df = main(filepath, target_column, log_dir, groupingparams, best_param)
    # ranking_df.to_csv(os.path.join(log_dir, 'ranking_df2.csv'))
    ranking_df = pd.read_csv('boruta_explog/ranking_df2.csv')
    plot_boruta(ranking_df, log_dir)
    plot_boruta_by_group(ranking_df, log_dir)