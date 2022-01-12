"""
Example: python benchmark_MOLI_drug.py

This script evaluates MOLI's predictive performance.
The results were not included in the formal benchmarking analysis since MOLI does not support custom datasets with a software package.
"""

from torch.utils.data import DataLoader

from MOLI import *
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import sys
import torch.optim as optim

parser = argparse.ArgumentParser(description='Benchmark MOLI')
parser.add_argument('--drug-file', type=str, default="")
parser.add_argument('--out', type=str, default="")
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--iter', type=int, default=1)
args = parser.parse_args()
print(args)

# load data
df_mutation = pd.read_csv("data/CCLE_mutations_processed_filtered.csv", index_col=0)
df_cna = pd.read_csv("data/CCLE_CN_processed_filtered.csv", index_col=0)
df_rna = pd.read_csv("data/CCLE_RNA_processed_filtered.csv", index_col=0)
df_protein = pd.read_csv("data/CCLE_protein_processed_filtered.csv", index_col=0).fillna(0)

drug_df = pd.read_csv(args.drug_file)
res_df = []

# train and test MOLI on one drug
def run_model(y_df, i):
    cv = KFold(n_splits=5, random_state=i, shuffle=True)
    res = []
    for train_cell, test_cell in cv.split(cell_lines):
        train_cell = cell_lines[train_cell]
        test_cell = cell_lines[test_cell]
        train_df = {"mutation": df_mutation.loc[train_cell, :],
                    "cna": df_cna.loc[train_cell, :],
                    "rna": df_rna.loc[train_cell, :],
                    "protein": df_rna.loc[train_cell, :]}
        test_df = {"mutation": df_mutation.loc[test_cell, :],
                   "cna": df_cna.loc[test_cell, :],
                   "rna": df_rna.loc[test_cell, :],
                   "protein": df_rna.loc[test_cell, :]}
        y_train = y_df[y_df.index.isin(train_cell)]
        y_test = y_df[y_df.index.isin(test_cell)]

        MOLI_df = {"mutation": df_mutation.shape[1],
                   "cna": df_cna.shape[1],
                   "rna": df_rna.shape[1],
                   "protein": df_rna.shape[1]}

        train_dataset = MultiOmicDataset(train_df, y_train['AUC'].values)
        test_dataset = MultiOmicDataset(test_df, y_test['AUC'].values)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
        model = MOLI(MOLI_df, 1024, 0.3, 1)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        train_res, test_res = train_loop(args.epochs, train_loader, test_loader, model, criterion, optimizer,
                                         task='regression')
        res.append(test_res[-1])
    return res

# loop through all the drugs
for drug_name in tqdm(drug_df['drug_name'].unique()):
    drug_curr_df = drug_df[drug_df['drug_name'] == drug_name].reset_index(drop=True)
    drug_response_map = drug_curr_df.set_index('DepMap_ID').to_dict()['AUC']
    y = df_mutation.index.map(drug_response_map).values
    cell_lines = df_mutation.index.values[~np.isnan(y)]
    y = y[~np.isnan(y)]
    y_df = pd.DataFrame(y, index=cell_lines, columns=['AUC'])
    tmp = {'drug_name': drug_name}
    res = Parallel(n_jobs=args.iter)(
        delayed(run_model)(y_df, i) for i in range(args.iter))
    torch.cuda.empty_cache()
    res = np.array(res).flatten()
    for i in range(res.size):
        tmp[f"cv{i}"] = res[i]
    tmp["avg"] = np.average(res)
    print(f"{drug_name}: {tmp['avg']:.6f}")
    sys.stdout.flush()
    res_df.append(tmp)

# save results
res_df = pd.DataFrame(res_df)
res_df.to_csv(args.out, index=False)
