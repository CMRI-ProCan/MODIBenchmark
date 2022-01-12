library(iClusterPlus)
library(optparse)
# R script to run iClusterPlus

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
mutation = readRDS("./data/CCLE_mutations_processed_filtered.rds")
cna = readRDS("./data/CCLE_CN_processed_filtered.rds")
rna = readRDS("./data/CCLE_RNA_processed_filtered.rds")
protein = readRDS("./data/CCLE_protein_processed_filtered.rds")
drug = read.csv("./data/broad_drug_processed_auc.csv")

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

# start running model
start_time <- Sys.time()
fit = iClusterPlus(dt1=mutation, dt2=cna, dt3=rna, dt4=protein, type = c("gaussian","gaussian","gaussian","gaussian"), K=30, 
                   alpha=c(1,1,1,1),lambda=c(0.03,0.03,0.03,0.03))
end_time <- Sys.time()
end_time - start_time

# save results
res = as.data.frame(fit$meanZ)
rownames(res) = common_samples

write.table(res, output, sep = ",", quote = F)
