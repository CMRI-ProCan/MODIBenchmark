"""
The script to run MOFA, following their original documentation
"""
from mofapy2.run.entry_point import entry_point
import pandas as pd
import sys
import time
import argparse

parser = argparse.ArgumentParser(description='Train MOFA')
parser.add_argument('--data-source', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--frac', type=float, default=1)
parser.add_argument('--gpu', default=False, action='store_true')
args = parser.parse_args()
print(args)

data_source = args.data_source
output = args.output
frac = args.frac

# load data
mutation = pd.read_pickle(f"{data_source}/CCLE_mutations_processed_filtered.pkl").reset_index().sample(frac=frac)
cn = pd.read_pickle(f"{data_source}/CCLE_CN_processed_filtered.pkl").reset_index().sample(frac=frac)
rna = pd.read_pickle(f"{data_source}/CCLE_RNA_processed_filtered.pkl").reset_index().sample(frac=frac)
protein = pd.read_pickle(f"{data_source}/CCLE_protein_processed_filtered.pkl").reset_index().sample(frac=frac)

mutation_mofa = mutation.melt(id_vars='DepMap_ID',
                              value_name='value',
                              var_name='feature')
mutation_mofa['view'] = 'mutation'
mutation_mofa['group'] = 'group_0'
mutation_mofa = mutation_mofa.rename(columns={'DepMap_ID': 'sample'})
mutation_mofa = mutation_mofa[['sample', 'group', 'feature', 'view', 'value']]

cn_mofa = cn.melt(id_vars='DepMap_ID',
                  value_name='value',
                  var_name='feature')
cn_mofa['view'] = 'cn'
cn_mofa['group'] = 'group_0'
cn_mofa = cn_mofa.rename(columns={'DepMap_ID': 'sample'})
cn_mofa = cn_mofa[['sample', 'group', 'feature', 'view', 'value']]

rna_mofa = rna.melt(id_vars='DepMap_ID',
                    value_name='value',
                    var_name='feature')
rna_mofa['view'] = 'rna'
rna_mofa['group'] = 'group_0'
rna_mofa = rna_mofa.rename(columns={'DepMap_ID': 'sample'})
rna_mofa = rna_mofa[['sample', 'group', 'feature', 'view', 'value']]

protein_mofa = protein.melt(id_vars='DepMap_ID',
                            value_name='value',
                            var_name='feature')
protein_mofa['view'] = 'protein'
protein_mofa['group'] = 'group_0'
protein_mofa = protein_mofa.rename(columns={'DepMap_ID': 'sample'})
protein_mofa = protein_mofa[['sample', 'group', 'feature', 'view', 'value']]

combined_mofa = pd.concat([mutation_mofa, cn_mofa, rna_mofa,
                           protein_mofa]).dropna().drop_duplicates(
    ["group", "view", "feature", "sample"])

start = time.time()
ent = entry_point()
ent.set_data_options(
    scale_groups=False,
    scale_views=True
)

ent.set_data_df(combined_mofa, likelihoods=["gaussian", "bernoulli", "gaussian", "gaussian"])

ent.set_model_options(
    factors=30,
    spikeslab_weights=True,
    ard_factors=True,
    ard_weights=True
)

ent.set_train_options(
    iter=200,
    convergence_mode="fast",
    startELBO=1,
    freqELBO=1,
    dropR2=0.001,
    gpu_mode=args.gpu,
    verbose=False,
    seed=1
)

ent.build()
ent.run()
elapsed_time = time.time() - start
print(f'MOFA time taken: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
ent.save(output)
