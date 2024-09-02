import pandas as pd 
from typing import Dict, List, Tuple
import json
import warnings
import numpy as np
import itertools
from loguru import logger
from utils import load_config


class FeatureDrivator:
    def __init__(self, list_of_features: list):
        self.features_for_derive = list_of_features
    def derive(self, df: pd.DataFrame):
        logger.info(f"Deriving features using {len(self.features_for_derive)} features")
        combinations = list(itertools.combinations(self.features_for_derive, 2))
        logger.info(f"Number of derived features: {len(combinations)}")
        # 对于每一对特征组合，生成派生变量并添加到df中
        for (feat1, feat2) in combinations:
            # 创建新的派生变量列名
            new_col_name = f'{feat1}_div_{feat2}'
            df[new_col_name] = df[feat1] / (df[feat2] + 1e-10)

        logger.info(f"Derived features: {df.columns[df.columns.str.contains('_div_')][:5]} ...")
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
        features_to_use = [feat for feat in df.columns if feat not in [self.target_column, 'sample_weight','agegroup', 'visitdurationgroup']]
        logger.info(f"""Filtering features using sorting method 
                    based on pick topn: {topn} in {len(features_to_use)} features 
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

        features_to_use = [feat for feat in self.features_list if feat in features_to_use]  + [feat for feat in features_to_use if feat not in self.features_list]
        features_to_use = features_to_use[:topn]
        logger.info(f"Selected features: {features_to_use[:5]} ...")
        df = df[features_to_use + [self.target_column, 'sample_weight','agegroup', 'visitdurationgroup']]
        return df


    def _selection(self, df: pd.DataFrame):
        features_to_use = [feat for feat in self.features_list if feat in df.columns]
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
        # dropna
        ## remove columns with all NaN values in CommonBloodTest
        namelist = [x[1] for x in self.ExaminationItemClass['CommonBloodTest']]
        result_cols2 = df.columns[df.columns.str.contains("|".join(namelist))]
        df = df.dropna(subset=result_cols2, how='all') 

        ## remove columns in results_col with only 1 unique value or all NaN
        result_cols = df.columns[df.columns.str.contains('Avg|Count|Sum|Max|Min|Median|Std|Skew|Kurt|Pct')]
        df = df.drop(columns=result_cols[df[result_cols].nunique() <= 1])

        ## remove columns with any NaN values
        other_cols = [col for col in df.columns if col not in result_cols]
        df = df.dropna(subset=other_cols, how='any')

        return df
    
    def _imputation(self, df: pd.DataFrame):
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
        # Group data by visitdurationgroup and calculate weights
        df['visitdurationgroup'] = pd.cut(df['VisitDuration'], 
                                          bins=[0, 42, 365, 1095, 1825,10000], 
                                          labels=["<6w", "6w-1y", "1-3y", "3-5y", "5y+"], ordered=True, right=False)
        df = df.dropna(subset=['visitdurationgroup'])
        weights = df['visitdurationgroup'].value_counts(normalize=True)
        df['sample_weight'] = df['visitdurationgroup'].map(lambda x: 1 / (weights[x] + 1e-10))
        return df
    
    def _subsetting(self, df: pd.DataFrame, pick_key: str):
        # subsetting data
        if pick_key != 'all':
            df = df[df['agegroup'] == pick_key]
        else:
            pass
        return df
    
    def _scalingX(self, df: pd.DataFrame):
        # scale the X data by min-max 
        result_cols = df.columns[df.columns.str.contains('Avg|Count|Sum|Max|Min|Median|Std|Skew|Kurt|Pct')]
        for col in result_cols:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df

    def _scalingY(self, df: pd.DataFrame, scale_factor, log_transform):
        # scale the Y data by scale_factor and log_transform
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