import pandas as pd
import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from best_params import opendb, get_best_params
from utils import plot_roc_summary

# todo 去除离群值

def within_n_std(data, column, n):
    return (data[column] - data[column].median()).abs() <= n * data[column].std()

# list all folder in repodir 
def LoadData(repodir):
        
    allfolder = [dir for dir in os.listdir(repodir) if os.path.isdir(os.path.join(repodir, dir))]
    dflist = [] 
    for folder in allfolder:
        labels_trained = [dir for dir in os.listdir(os.path.join(repodir, folder)) if os.path.isdir(os.path.join(repodir, folder, dir))]
        for label in labels_trained:
            df_path = os.path.join(repodir, folder, label, "result","0",'feature_importance.csv')
            df = pd.read_csv(df_path)
            df['group'] = label
            df['param_id'] = folder
            df['sequence_ids'] = '0'
            dflist.append(df)

    df = pd.concat(dflist, axis=0)
    return df, labels_trained

# plot a verticle boxplot with dots x is Importance y is Feature for each group

def LoadDataFromNNI(nnidir, metric_to_optimize: list=[('default','minimize')], number_of_trials=25):
    best_db_path = os.path.join(nnidir, 'db','nni.sqlite')
    df = opendb(best_db_path)
    ls_of_params = get_best_params(df, metric_to_optimize, number_of_trials)
    
    dflist = []
    for param_id, _, sequence_ids in ls_of_params:
        for sequence_id in sequence_ids:
            print(sequence_id)
            bestparamfolder = os.path.join(nnidir, 'result', str(sequence_id))
            dfpath = os.path.join(bestparamfolder, 'feature_importance.csv')
            df = pd.read_csv(dfpath)
            df['sequence_ids'] = sequence_id
            df['param_id'] = param_id
            df['group'] = 'all'
            dflist.append(df)
    df = pd.concat(dflist, axis=0)
    return df, ['all']

def PlotImportance(df, labels, outdir,n=15):
    # rename column featuer to Feature if feature in df.columns
    if 'feature' in df.columns:
        df.rename(columns={'feature': 'Feature'}, inplace=True)
    if 'weight' in df.columns:
        df.rename(columns={'weight': 'Importance'}, inplace=True)

    mapcategory2feature = {}    
    unique_feature = df['Feature'].unique().tolist()
    for key in ExaminationItemClass.keys():
        namelist = [x[1] for x in ExaminationItemClass[key]]
        pattern = '|'.join(namelist)
        mapcategory2feature[key] = [x for x in unique_feature if re.search(pattern, x)]
    mapcategory2feature['Clinical'] = ['FirstVisitAge','CIndU','Gender']
    mapcategory2feature['DerivativeVariables'] = [x for x in unique_feature if '_div_' in x]

    df['Feature_group'] = df['Feature'].map({v: k for k in mapcategory2feature for v in mapcategory2feature[k]})
    
    # map categoty to color
    colors = sns.color_palette("colorblind", len(mapcategory2feature.keys()))
    mapcategory2color = dict(zip(mapcategory2feature.keys(), colors))
    top_n_list = []
    for label in labels:
        print(label)    
        data = df[df['group'] == label]
        #  for each feature remove outliers by mean +- n*std
        filtereddata = []
        for feature in data['Feature'].unique():
            temp = data[data['Feature'] == feature]
            temp.reset_index(drop=True, inplace=True)
            filtereddata.append(temp[within_n_std(temp, 'Importance', 1)])
        data = pd.concat(filtereddata, axis=0, ignore_index=True)
        # importance mean topn
        feature_to_plot = data.groupby('Feature')['Importance'].median().sort_values(ascending=False).index[:n].tolist()
        
        top_n_list.append({'label': label, 
                           'n': n,
                           'top_n': feature_to_plot})
        data = data[data['Feature'].isin(feature_to_plot)]   
        data['Feature'] = pd.Categorical(data['Feature'], categories=feature_to_plot, ordered=True)
        data = data.sort_values('Feature')

        # 创建一个图形和轴对象
        fig, ax = plt.subplots(figsize=(8, 8)) # figsize=(width, height) in inches

        # 设置轴对象的位置和大小
        ax.set_position([0.5, 0.1, 0.45, 0.8])  # [left, bottom, width, height] in figure coordinates

        # 自定义中位线和误差线
        # boxprops = dict(linestyle='-', linewidth=2, color='blue')  # 盒子的样式
        medianprops = dict(linestyle='-', linewidth=3, color='black')  # 中位数线的样式
        whiskerprops = dict(linestyle='--', linewidth=2, color='black')  # 误差线的样式
        # boxplot with dots
        sns.boxplot(data , x = 'Importance', y = 'Feature', 
                    hue='Feature_group',  
                    palette=mapcategory2color,
                    ax = ax,
                    width = 0.5, 
                    # boxprops=boxprops, 
                    medianprops=medianprops, whiskerprops=whiskerprops)
        sns.stripplot(data, x = 'Importance', y = 'Feature', color='orange',
                    size=5, jitter=0.25, ax = ax, marker='o', 
                    )


        plt.title('Feature Importance in ' + label, fontsize=20,pad=15)
        plt.xlabel('Importance', fontsize=15)
        plt.xlim(0, max(data['Importance']) + 10)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylabel('Feature', fontsize=15)
        plt.savefig(os.path.join(outdir, label + '_feature_importance.png'))
        plt.close()

    return top_n_list

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Plot feature importance')
    parser.add_argument('--repodir', type=str, default = None, help='repo directory')
    parser.add_argument('--nnidir', type=str, default = None, help='nni directory')
    parser.add_argument('--metric', type=str, default = None, help='metric to optimize')
    parser.add_argument('--minimize', type=bool, default = None, help='minimize or maximize')
    parser.add_argument('--number_of_trials', type=int, default = None, help='number of trials')
    return parser.parse_args()

if __name__ == '__main__':
    with open ('ExaminationItemClass_ID.json', 'r') as json_file:
        ExaminationItemClass = json.load(json_file)
    args = argparser()
    repodir = args.repodir
    nnidir = args.nnidir
    

    # RefreshFeatureImportance(repodir)
    if repodir:
        outdir = os.path.join('VariablesImportance', os.path.basename(repodir))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        
        # plot ROC Summary
        jsonfilepath = [p for p in os.listdir(repodir) if p.endswith('_results.json')][0]
        if os.path.exists(os.path.join(repodir, jsonfilepath)):
            df, labels = LoadData(repodir)
            plot_roc_summary(os.path.join(repodir, jsonfilepath), outdir)
        else:
            print(f"ROC summary file {jsonfilepath} not found")
            df = None
            labels = None
            

        
    elif nnidir:
        nnidir = os.path.realpath(nnidir) 
        outdir = os.path.join('VariablesImportance', f"{os.path.basename(nnidir)}_{args.metric}_top{args.number_of_trials}_fromnni")
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        m = 'minimize' if args.minimize else 'maximize'
        df, labels = LoadDataFromNNI(nnidir, metric_to_optimize=[(args.metric, m)], number_of_trials=args.number_of_trials)

    if df is not None:
        top_n_list = PlotImportance(df, labels, outdir)
        df.to_csv(os.path.join(outdir, 'feature_importance_summary.csv'), index=False)
        with open(os.path.join(outdir, 'feature_importance_summary.json'), 'w') as f:
            json.dump(top_n_list, f, indent=4)
            