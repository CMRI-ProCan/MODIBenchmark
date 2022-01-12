# the shell script to run iClusterPlus and iClusterBayes with different sample sizes

Rscript R/iClusterPlus.R --frac 0.2 --out data/drug/iCluster/iClusterPlus_frac0.2.csv | tee logs/iClusterPlus_frac0.2.log
Rscript R/iClusterPlus.R --frac 0.4 --out data/drug/iCluster/iClusterPlus_frac0.4.csv | tee logs/iClusterPlus_frac0.4.log
Rscript R/iClusterPlus.R --frac 0.6 --out data/drug/iCluster/iClusterPlus_frac0.6.csv | tee logs/iClusterPlus_frac0.6.log
Rscript R/iClusterPlus.R --frac 0.8 --out data/drug/iCluster/iClusterPlus_frac0.8.csv | tee logs/iClusterPlus_frac0.8.log
Rscript R/iClusterPlus.R --frac 1 --out data/drug/iCluster/iClusterPlus_frac1.csv | tee logs/iClusterPlus_frac1.log

#Rscript R/iClusterBayes.R --frac 0.2 --out data/drug/iCluster/iClusterBayes_frac0.2.csv | tee logs/iClusterBayes_frac0.2.log
#Rscript R/iClusterBayes.R --frac 0.4 --out data/drug/iCluster/iClusterBayes_frac0.4.csv | tee logs/iClusterBayes_frac0.4.log
#Rscript R/iClusterBayes.R --frac 0.6 --out data/drug/iCluster/iClusterBayes_frac0.6.csv | tee logs/iClusterBayes_frac0.6.log
#Rscript R/iClusterBayes.R --frac 0.8 --out data/drug/iCluster/iClusterBayes_frac0.8.csv | tee logs/iClusterBayes_frac0.8.log
#Rscript R/iClusterBayes.R --frac 1 --out data/drug/iCluster/iClusterBayes_frac1.csv | tee logs/iClusterBayes_frac1.log
