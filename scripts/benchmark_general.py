"""
Example: python scripts/benchmark_general.py --data-file data/drug/PCA.csv --drug-file data/broad_drug_processed_auc.csv --out result_files/PCA_MAE.csv | tee logs/PCA_MAE.log
This script runs random forest as the downstream algorith for general-purpose integration methods to predict drug response.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import numpy as np
import argparse
from tqdm import tqdm
from joblib import delayed, Parallel
import sys

parser = argparse.ArgumentParser(description='Benchmark general')
parser.add_argument('--data-file', type=str, default="")
parser.add_argument('--drug-file', type=str, default="")
parser.add_argument('--out', type=str, default="")
args = parser.parse_args()

# load data
input_df = pd.read_csv(args.data_file, index_col=0)
drug_df = pd.read_csv(args.drug_file)

res_df = []
# loop through all drugs
for drug_name in tqdm(drug_df['drug_name'].unique()):
    drug_curr_df = drug_df[drug_df['drug_name'] == drug_name].reset_index(drop=True)
    drug_response_map = drug_curr_df.set_index('DepMap_ID').to_dict()['AUC']
    X = SimpleImputer().fit_transform(input_df.values)
    y = input_df.index.map(drug_response_map).values
    X = X[~np.isnan(y), :]
    y = y[~np.isnan(y)]
    model = RandomForestRegressor(max_features='sqrt')
    res = Parallel(n_jobs=100)(
        delayed(cross_val_score)(model, X, y, cv=5, n_jobs=1, scoring='r2') for i in
        range(100))
    res = np.array(res).flatten()
    tmp = {'drug_name': drug_name}
    for i in range(len(res)):
        tmp[f"cv{i}"] = res[i]
    tmp['avg'] = np.average(res)
    print(f"{drug_name}: {tmp['avg']:.6f}")
    sys.stdout.flush()
    res_df.append(tmp)

# save results
res_df = pd.DataFrame(res_df)
res_df.to_csv(args.out, index=False)
