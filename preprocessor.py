import pandas as pd 
from typing import Dict, List, Tuple
import json
import warnings
import numpy as np
import itertools
from loguru import logger
from utils import load_config
from pandas.errors import PerformanceWarning
def get_asso_feat(feat, featlist):
    '''
    feat: str, feature to be associated
    featlist: list of str, list of features to be associated with feat
    '''
    assofeat = [f for f in featlist if feat in f]
    return assofeat

class FeatureDrivator:
    def __init__(self, list_of_features: list):
        self.features_for_derive = list_of_features

    def derive(self, df: pd.DataFrame):
        features_to_derive = []
        for feat in self.features_for_derive:
            assofeat = get_asso_feat(feat, df.columns)
            features_to_derive.extend(assofeat)
        features_to_derive = list(set(features_to_derive))
        logger.info(f"Deriving features using {len(features_to_derive)} features")
        combinations = list(itertools.combinations(features_to_derive, 2))
        logger.info(f"Number of derived features: {len(combinations)}")
        # 对于每一对特征组合，生成派生变量并添加到df中
        # closewarning # PerformanceWarning: DataFrame is highly fragmented. 
        # warnings.filterwarnings("ignore", category=PerformanceWarning, message="DataFrame is highly fragmented.")
        # for (feat1, feat2) in combinations:
        #     # 创建新的派生变量列名
        #     new_col_name = f'{feat1}_div_{feat2}'
        #     df[new_col_name] = df[feat1] / (df[feat2] + 1e-10)
        # warnings.filterwarnings("default", category=PerformanceWarning, message="DataFrame is highly fragmented.")

        # 预分配内存
        new_features = np.empty((len(df), len(combinations)))
        new_feature_names = []
        # 使用numpy进行批量计算
        for i, (feat1, feat2) in enumerate(combinations):
            new_features[:, i] = df[feat1].values / (df[feat2].values + 1e-10)
            new_feature_names.append(f'{feat1}_div_{feat2}')
        # 一次性添加所有新特征
        # print value count of new_feature_names if value count is greater than 1, print

        df = pd.concat([df, pd.DataFrame(new_features, columns=new_feature_names, index=df.index)], axis=1)

        logger.info(f"Derived features: {df.columns[df.columns.str.contains('_div_')][:5]} ...")
        logger.info(f"duplicated column names: {df.columns[df.columns.duplicated()]}")

        return df

class FeatureFilter:
    def __init__(self, 
                 target_column: str,
                 method: str, 
                 features_list: list):
        '''
        method: str, 
        features_list: list|None = None):
        when method is sorting, features_list is used as a list of features sorted by importance
        when method is selection, features_list is used as a list of features to be selected
        '''
        self.target_column = target_column
        self.method = method
        if method not in ['sorting', 'selection']:
            raise ValueError("Invalid method")
        self.features_list = features_list

    def filter(self, df: pd.DataFrame, **kwargs):
        if self.method == 'sorting':
            topn = kwargs.get('topn', None)
            return self._sorting(df, topn)
        elif self.method == 'selection':
            return self._selection(df)
    
    def _sorting(self, df: pd.DataFrame, topn: int|float|None):
        '''
        topn: int|float, 
        if topn is a positive integer, it is used as the number of features to be selected
        if topn is a float between 0 and 1, it is used as the ratio of features to be selected
        if topn is a negative integer, it means all features are selected
        if topn is None, all features are selected
        '''
        # feature filtering by importance sorting
        original_features_to_use = [feat for feat in df.columns if feat not in [self.target_column, 'sample_weight','agegroup', 'visitdurationgroup']]
        logger.info(f"""Filtering features using sorting method 
                    based on pick topn: {topn} in {len(original_features_to_use)} features 
                    in the order of importance provided: {self.features_list[:5]} ...""")
        if topn is None: # only when topn is not provided in search space will it be None
            raise ValueError("topn is not found in search space while FeatureFilter is instanced pls check")
        # search space topn is either a list of int greater than 1 with -1 suggesting using all features or a float between 0 and 1 with 1 suggesting using all features
        if topn > 1: # topn is an integer
            topn = topn
        elif topn < 0: # topn is -1 use all features
            topn = len(features_to_use)
        else: # topn is a ratio between 0 and 1
            topn = round(len(features_to_use) * topn) 
        # rank features to use by sorted_features if not in features_to_use then place at the end
        features_to_use = []
        for feat in self.features_list: # 遍历features_list 在featuretouse 中添加关联特征 
            assofeat = get_asso_feat(feat, df.columns)
            features_to_use.extend(assofeat)

        features_to_use.extend([feat for feat in original_features_to_use if feat not in features_to_use]) # 在featuretouse 结尾添加 其余特征

        features_to_use = features_to_use[:topn]
        logger.info(f"Selected features: {features_to_use[:5]} ...")
        df = df[features_to_use + [self.target_column, 'sample_weight','agegroup', 'visitdurationgroup']]
        return df

  

    def _selection(self, df: pd.DataFrame):
        features_to_use = []
        for feat in self.features_list:
            assofeat = get_asso_feat(feat, df.columns)
            features_to_use.extend(assofeat)

        logger.info(f"""Filtering features using selection method based on features provided: 
                    {self.features_list[:5]} ...
                    provided features number: {len(self.features_list)}
                    selected features number: {len(features_to_use)}
                    """)
        df = df[features_to_use + [self.target_column, 'sample_weight','agegroup', 'visitdurationgroup']]
        return df

class Preprocessor:
    def __init__(self, 
                 target_column: str,
                 groupingparams: Dict[str, List[str]],
                 feature_reference: str = 'ExaminationItemClass_ID.json',
                 feature_derive: FeatureDrivator|None = None, 
                 FeaturFilter: FeatureFilter|None = None
                 ):
        self.target_column = target_column
        self.ExaminationItemClass = json.load(open(feature_reference, 'r'))
        self.groupingparams = groupingparams
        self.feature_derive = feature_derive
        self.feature_filter = FeaturFilter


    def _dropna(self, df: pd.DataFrame):
        logger.info(f"Dropping NaN values")
        # dropna
        ## remove columns with all NaN values in CommonBloodTest
        namelist = [x[1] for x in self.ExaminationItemClass['CommonBloodTest']]
        result_cols2 = df.columns[df.columns.str.contains("|".join(namelist))]
        df = df.dropna(subset=result_cols2, how='all') 

        ## remove columns in results_col with only 1 unique value or all NaN
        result_cols = df.columns[df.columns.str.contains('Avg|Count|Sum|Max|Min|Median|Std|Skew|Kurt|Pct')]
        df = df.drop(columns=result_cols[df[result_cols].nunique() <= 1])

        ## remove other columns with any NaN values
        other_cols = [col for col in df.columns if col not in result_cols]
        df = df.dropna(subset= other_cols, how='any')
        df = df.dropna(subset = [self.target_column], how='any')

        return df
    
    def _imputation(self, df: pd.DataFrame):
        logger.info(f"Imputing missing values")
        # group data by agegroup
        bins = self.groupingparams['bins']
        labels = self.groupingparams['labels']
        df = df.assign(agegroup=pd.cut(df['FirstVisitAge'], bins= bins, labels= labels, right=False))
        df['agegroup'] = pd.Categorical(df['agegroup'], categories= labels, ordered=True)
       
        # Missing value imputation based on agegroup
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        result_cols = df.columns[df.columns.str.contains('Avg|Count|Sum|Max|Min|Median|Std|Skew|Kurt|Pct')]
        for col in result_cols:
            overall_median = df[col].median()
            df[col] = df.groupby('agegroup', observed=True)[col].transform(
                lambda x: x.fillna(x.median() if not np.isnan(x.median()) else overall_median)
            )
        warnings.filterwarnings("default", category=RuntimeWarning, message="Mean of empty slice")
        return df
    
    def _weighting(self, df: pd.DataFrame):
        logger.info(f"Weighting data by visitdurationgroup")
        # Group data by visitdurationgroup and calculate weights
        # warnings.filterwarnings("ignore", category=PerformanceWarning, message="DataFrame is highly fragmented")
        # df['visitdurationgroup'] = pd.cut(df['VisitDuration'], 
        #                                   bins=[0, 42, 365, 1095, 1825,10000], 
        #                                   labels=["<6w", "6w-1y", "1-3y", "3-5y", "5y+"], ordered=True, right=False)
        # warnings.filterwarnings("default", category=PerformanceWarning, message="DataFrame is highly fragmented")

        # weights = df['visitdurationgroup'].value_counts(normalize=True)
        # df['sample_weight'] = df['visitdurationgroup'].map(lambda x: 1 / (weights[x] + 1e-10))
        # 使用 numpy 进行分组计算
        bins = [0, 42, 365, 1095, 1825,10000]
        labels = ["<6w", "6w-1y", "1-3y", "3-5y", "5y+"]

        # 将 VisitDuration 转换为 numpy 数组
        visit_duration = df['VisitDuration'].values
        
        # 使用 numpy 的 digitize 函数进行分组
        try:
            group_indices = np.digitize(visit_duration, bins, right=False)
            # unique, counts = np.unique(group_indices, return_counts=True)
            # logger.info(f"value count of groupindices: {dict(zip(unique, counts))}")

        except Exception as e:
            logger.error(f"Error in digitize: {e}")
            # print the visit_duration of which group is NaN
            nan_indices = np.isnan(visit_duration)
            nan_visit_duration = visit_duration[nan_indices]
            logger.error(f"NaN VisitDuration: {nan_visit_duration}")
            raise e
        
        # 创建 visitdurationgroup 列
        df['visitdurationgroup'] = pd.Categorical.from_codes(group_indices - 1, categories=labels, ordered=True)
        logger.info(df['visitdurationgroup'].value_counts())
        # 计算权重
        value_counts = df['visitdurationgroup'].value_counts(normalize=True)
        df['sample_weight'] = df['visitdurationgroup'].map(lambda x: 1 / (value_counts[x] + 1e-10))
        logger.info(f"sample_weight for each visitdurationgroup: {df['sample_weight'].value_counts()}")
        # print duplicated column names in df
        logger.info(f"duplicated column names: {df.columns[df.columns.duplicated()]}")
        return df
    
    def _subsetting(self, df: pd.DataFrame, pick_key: str):
        # subsetting data
        logger.info(f"Subsetting data by agegroup: {pick_key}")
        if pick_key != 'all':
            df = df[df['agegroup'] == pick_key]
        else:
            pass
        return df
    
    def _scalingX(self, df: pd.DataFrame):
        # scale the X data by min-max 
        logger.info(f"Scaling the X data by min-max")
        result_cols = df.columns[df.columns.str.contains('Avg|Count|Sum|Max|Min|Median|Std|Skew|Kurt|Pct')]
        for col in result_cols:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df

    def _scalingY(self, df: pd.DataFrame, scale_factor, log_transform):
        # scale the Y data by scale_factor and log_transform
        logger.info(f"Scaling the Y data by scale_factor: {scale_factor} and log_transform: {log_transform}")
        df[self.target_column] = np.round(df[self.target_column] / scale_factor) + 1
        if log_transform == "log2":
            df[self.target_column] = np.log2(df[self.target_column])
        elif log_transform == "log10":
            df[self.target_column] = np.log10(df[self.target_column])
        else:
            pass
        return df
    
    def preprocess(self, df: pd.DataFrame, 
                    scale_factor: int,
                    log_transform: str, 
                    pick_key = '0-2',
                    topn: int|float|None = None):
        # dropna
        df = self._dropna(df)

        # Missing value imputation
        df = self._imputation(df)

        # Group data by visitdurationgroup and calculate weights
        df = self._weighting(df)
        
        # subsetting data
        df = self._subsetting(df, pick_key)

        # scale the X data by min-max 
        df = self._scalingX(df)

        # scale the Y data by scale_factor and log_transform
        df = self._scalingY(df, scale_factor, log_transform)

        # feature derivation if specified
        if self.feature_derive is not None:
            df = self.feature_derive.derive(df)

        # feature filtering if specified
        if self.feature_filter is not None:
            # when filtration is instanced, topn is usually provided as int
            # when no topn in search space topn is None if the filterer is instanced it will trigger error
            df = self.feature_filter.filter(df, topn=topn)
        logger.info(f"Preprocessed data head: {df.head()}")
        
        X = df.drop(columns=[self.target_column, 'sample_weight','agegroup', 'visitdurationgroup'])
        y = df[self.target_column]
        sample_weight = df['sample_weight']

        logger.info(f"Preprocessed data shape: {X.shape}, {y.shape}")
        logger.info(f"NaN in X: {X.isna().sum().sum()}")
        logger.info(f"NaN in y: {y.isna().sum()}")
        logger.info(f"NaN in sample_weight: {sample_weight.isna().sum()}")
        return X, y, sample_weight

if __name__ == "__main__":
    df = pd.read_csv('output/dataforxgboost_ac.csv')
    groupingparams = load_config('groupingsetting.yml')['groupingparams']
    pp = Preprocessor(target_column='VisitDuration', groupingparams=groupingparams)
    X, y, sample_weight = pp.preprocess(df)
    print(X.head())
    print(y.head())
    print(sample_weight.head())