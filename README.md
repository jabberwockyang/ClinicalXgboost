# ClinicalXgboost

一个基于xgboost探索预测病程相关检验指标的项目

此处主要同步xgboost模型训练相关代码

## 项目主要步骤

1. 数据 sqlite db搭建
2. 数据探索性分析
3. 机器学习模型训练

## 数据库搭建

## 数据探索性分析

## 机器学习模型训练
1. 结合nni超参数调优，进行xgboost模型训练，选取最优模型，明确最重要的15个变量

    ```bash
    conda activate nni
    cd ClinicalXgboost
    nnictl create --config config_nni1.yml --port 8081 # nnictl resume {exp_id} --port 8081
    python3 importance.py --nnidir nni1_explog/{exp_id} --metric default --minimize True --number_of_trials 7
    ```

2. 对15个变量进行特征工程，构建派生变量 加入派生变量后，结合nni超参数调优，进行xgboost模型训练
    
    ```bash
    conda activate nni
    cd ClinicalXgboost
    nnictl create --config config_nni2.yml --port 8081 # nnictl resume {exp_id} --port 8081
    python3 importance.py --nnidir nni2_explog/{exp_id} --metric default --minimize True --number_of_trials 7
    ```

3. 选取最优模型，评估模型性能，描述变量重要性