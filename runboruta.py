
import pandas as pd
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
import seaborn as sns
from utils import load_data, load_config, augment_samples, load_feature_list_from_boruta_file
from best_params import opendb, get_best_params
from preprocessor import Preprocessor, FeatureDrivator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import queue
import argparse
import uuid

class PoisonPill:
    pass

def main(filepath,  params, preprocessor,logdir):
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
                                   random_state=i, max_iter=10)
    
        boruta_selector.fit(X_train, y_train)
        confirmed_vars = X.columns[boruta_selector.support_]
        feature_ranks = boruta_selector.ranking_
        logger.info(f"Iteration {i+1} finished")
        output_queue.put((i+1, confirmed_vars, feature_ranks))
        logger.info(f"Iteration {i+1} put into queue")

    def get_and_write(output_queue):
        while True:
            item = output_queue.get()
            if isinstance(item, PoisonPill):
                logger.info(f"PoisonPill received, exiting...")
                return
            elif isinstance(item, tuple):
                i, confirmed_vars, feature_ranks = item
                ranking_df.loc[i] = feature_ranks
                ranking_df.to_csv(os.path.join(log_dir,'ranking_df.csv'))
                
                with open(os.path.join(log_dir, 'confirmed_vars.txt'), 'a') as f:
                    f.write(f"{i},{','.join(confirmed_vars)}\n")
            else:
                raise ValueError(f"Invalid item type: {type(item)}")

    # 使用多线程执行 Boruta
    with ThreadPoolExecutor(max_workers=6) as executor:
        consumer = executor.submit(get_and_write, output_queue)
        producers = [executor.submit(run_boruta, X, y, i) for i in range(20)]
        for future in producers:
            future.result()  # 等待所有线程完成
        logger.info(f"All threads finished, sending PoisonPill...")
        output_queue.put(PoisonPill())
        logger.info(f"PoisonPill sent")

        re = consumer.result()
        logger.info(f"Consumer result: {re}")


def plot_boruta(ranking_df,log_dir, name = 'boruta'):
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


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str)
    parser.add_argument('--best_db_path', type=str)

    parser.add_argument('--target_column', type=str, default='VisitDuration')
    parser.add_argument('--log_dir', type=str, default='boruta_explog')
    parser.add_argument('--groupingparams', type=str, default='groupingsetting.yml')
    parser.add_argument('--features_for_derivation', type=str, default=None, help='a file with features for derivation')
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    filepath = args.filepath
    target_column = args.target_column
    log_dir = args.log_dir
    # generate unique experiment name
    experiment_name = str(uuid.uuid4())
    logger.info(f"Experiment name: {experiment_name}")
    log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    # save copy of args
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        f.write(str(args))

    groupingparams = load_config(args.groupingparams)['groupingparams']

    best_db_path = args.best_db_path
    df  = opendb(best_db_path)
    ls_of_params = get_best_params(df, [('default','minimize')], 1)
    best_param = ls_of_params[0][1]
    # save copy of best_param
    with open(os.path.join(log_dir, 'best_param.txt'), 'w') as f:
        f.write(str(best_param))
    
    # 实例化特征衍生
    features_for_deri = load_feature_list_from_boruta_file(args.features_for_derivation) if args.features_for_derivation else None
    fd = FeatureDrivator(features_for_deri) if features_for_deri else None
    
    # 实例化预处理器
    preprocessor = Preprocessor(target_column,
                                 groupingparams,
                                 feature_derive = fd)

    main(filepath, best_param, preprocessor, log_dir)

    ranking_df = pd.read_csv(os.path.join(log_dir, 'ranking_df.csv'))
    plot_boruta(ranking_df, log_dir)
    plot_boruta_by_group(ranking_df, log_dir)