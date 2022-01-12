"""
Example: python scripts/benchmark_cancer_type.py --sample-info data/sample_info.csv --out result_files/cancer_type.csv
This script runs random forest as the downstream algorith for general-purpose integration methods to predict cancer type.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
import numpy as np
import argparse
from joblib import delayed, Parallel
import sys

parser = argparse.ArgumentParser(description='Benchmark general')
parser.add_argument('--sample-info', type=str, default="")
parser.add_argument('--out', type=str, default="")

args = parser.parse_args()

input_files = {"EarlyConcatenation": "/home/scai/MultiOmicReview/data/ccle_early_concat.csv.gz",
               "PCA": "/home/scai/MultiOmicReview/data/drug/PCA.csv",
               "UMAP": "/home/scai/MultiOmicReview/data/drug/UMAP.csv",
               "MOFA": "/home/scai/MultiOmicReview/data/drug/MOFA/mofa_30factor_frac1.0.csv",
               "DIABLO": "/home/scai/MultiOmicReview/data/drug/DIABLO/diablo_30factors.csv",
               "iClusterPlus": "/home/scai/MultiOmicReview/data/drug/iCluster/iClusterPlus_frac1.csv",
               "iClusterBayes": "/home/scai/MultiOmicReview/data/drug/iCluster/iClusterBayes_frac1.csv",
               "moCluster": "/home/scai/MultiOmicReview/data/drug/moCluster/moCluster_30factor_frac1.csv"}

res_df = []
# loop through different methods
for key in input_files.keys():
    input_df = pd.read_csv(input_files[key], index_col=0)
    sample_info = pd.read_csv(args.sample_info).set_index("DepMap_ID")
    sample_info = sample_info[sample_info.index.isin(input_df.index.values)]
    type_count = sample_info.groupby(['primary_disease']).size()
    selected_type = type_count[type_count >= 10].index.values

    selected_samples = sample_info[sample_info['primary_disease'].isin(selected_type)].index.values
    sample_info = sample_info[sample_info.index.isin(selected_samples)]
    input_df = input_df[input_df.index.isin(selected_samples)]

    cancer_type_map = sample_info.to_dict()['primary_disease']
    X = SimpleImputer().fit_transform(input_df.values)
    y = input_df.index.map(cancer_type_map).values

    model = RandomForestClassifier()
    res = Parallel(n_jobs=100)(
        delayed(cross_val_score)(model, X, y, cv=StratifiedKFold(shuffle=True, random_state=i), n_jobs=1,
                                 scoring='accuracy') for i in
        range(100))
    res = np.array(res).flatten()
    print(f"{key} Accuracy: {np.average(res):.6f}")
    sys.stdout.flush()

    tmp = {'model': key}
    for i in range(len(res)):
        tmp[f"cv{i}"] = res[i]
    tmp['avg'] = np.average(res)
    res_df.append(tmp)

# save results
res_df = pd.DataFrame(res_df)
res_df.to_csv(args.out, index=False)
