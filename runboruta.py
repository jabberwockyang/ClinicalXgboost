
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import seaborn as sns
from utils import preprocess_data, load_data, load_config, augment_samples

def main(filepath, target_column, log_dir,groupingparams):
    data = load_data(filepath)
    scale_factor = 2
    log_transform = "log2"
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
    # 初始化随机森林模型
    rf = RandomForestRegressor(n_jobs=6, max_depth=5, random_state=42)

    # 初始化存储特征排名的 DataFrame
    ranking_df = pd.DataFrame(columns=X_train.columns)

    # 运行 Boruta 20 次
    for i in range(10):
        print(f"Iteration {i+1}")
        
        # 初始化Boruta特征选择器
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=i, max_iter=50)
        
        # 对训练数据进行特征选择
        boruta_selector.fit(X_train.values, y_train.values)
        
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
    ranking_df = main(filepath, target_column, log_dir, groupingparams)
    ranking_df.to_csv(os.path.join(log_dir, 'ranking_df.csv'))
    plot_boruta(ranking_df, log_dir)