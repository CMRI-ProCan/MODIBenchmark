# script to run moCluster
library(mogsa)
library(optparse)


option_list = list(
  make_option("--frac", type="double", default=1.0, metavar="fraction"),
  make_option("--out", type="character", default="",  metavar="character")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)
opt
fraction = opt$frac
output = opt$out

# load data
mutation = readRDS("../data/CCLE_mutations_processed_filtered.rds")
cna = readRDS("../data/CCLE_CN_processed_filtered.rds")
rna = readRDS("../data/CCLE_RNA_processed_filtered.rds")
protein = readRDS("../data/CCLE_protein_processed_filtered.rds")
drug = read.csv("../data/broad_drug_processed_auc.csv")

common_samples = intersect(intersect(rownames(mutation), rownames(cna)),rownames(rna))
common_samples = intersect(common_samples, rownames(protein))
common_samples = sample(common_samples, fraction*length(common_samples))

mutation = mutation[common_samples,]
cna = cna[common_samples,]
rna = rna[common_samples,]
protein = protein[common_samples,]

mutation <- sapply(mutation, as.integer)
cna <- sapply(cna, as.numeric)
rna <- sapply(rna, as.numeric)
protein <- sapply(protein, as.numeric)
protein[is.na(protein)] = 0

mo.combined = list(t(mutation), t(cna), t(rna), t(protein))

# start running model
start_time <- Sys.time()
moa <- mbpca(mo.combined, ncomp = 30, k = "all", method = "globalScore", 
             option = "lambda1", center=TRUE, scale=FALSE, moa = TRUE, 
             svd.solver = "fast", maxiter = 1000)
end_time <- Sys.time()
end_time - start_time

# save results
res.df = as.data.frame(moa@fac.scr)
rownames(res.df) = common_samples
write.table(res.df, "../data/drug/moCluster/moCluster_30factor_frac1.csv", sep = ",", quote = F)