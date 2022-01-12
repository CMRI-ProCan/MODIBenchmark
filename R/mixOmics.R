library(mixOmics)

# load data
# mutation = read.csv("../data/CCLE_mutations_processed_filtered.csv", row.names = 1)
# saveRDS(mutation, "../data/CCLE_mutations_processed_filtered.rds")
# 
# cna = read.csv("../data/CCLE_CN_processed_filtered.csv", row.names = 1)
# saveRDS(cna, "../data/CCLE_CN_processed_filtered.rds")
# 
# rna = read.csv("../data/CCLE_RNA_processed_filtered.csv", row.names = 1)
# saveRDS(rna, "../data/CCLE_RNA_processed_filtered.rds")
# 
# protein = read.csv("../data/CCLE_protein_processed_filtered.csv", row.names = 1)
# saveRDS(protein, "../data/CCLE_protein_processed_filtered.rds")

# load data from RDS
mutation = readRDS("../data/CCLE_mutations_processed_filtered.rds")
cna = readRDS("../data/CCLE_CN_processed_filtered.rds")
rna = readRDS("../data/CCLE_RNA_processed_filtered.rds")
protein = readRDS("../data/CCLE_protein_processed_filtered.rds")
sample_info = read.csv("../data/sample_info.csv", row.names = 1)

common_samples = intersect(intersect(rownames(mutation), rownames(cna)),rownames(rna))

common_samples = sample(common_samples, 0.2*length(common_samples))
common_samples = common_samples[1:3]
mutation = mutation[common_samples,]
cna = cna[common_samples,]
rna = rna[common_samples,]
protein = protein[common_samples,]
Y = sample_info[common_samples, "lineage"]

data = list(mutation = mutation, 
            CNV = cna,
            RNA = rna,
            protein = protein)
# check dimension
lapply(data, dim)

design = matrix(0.1, ncol = length(data), nrow = length(data), 
                dimnames = list(names(data), names(data)))
diag(design) = 0

# run model
start_time <- Sys.time()
sgccda.res = block.splsda(X = data, Y = Y, ncomp = 30, 
                          design = design)
end_time <- Sys.time()
end_time - start_time

# save results
res = as.data.frame(cbind(sgccda.res$variates$mutation, sgccda.res$variates$CNV, sgccda.res$variates$RNA, sgccda.res$variates$protein))
write.table(res, "../data/drug/DIABLO/diablo_30factors.csv", sep=",", quote = F)