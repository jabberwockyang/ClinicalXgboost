import pandas as pd

df = pd.read_csv('/root/ClinicalXgboost/nni5_explog/lL1naWEh/datapreprocessed.csv')

print(df['target'].head(25))