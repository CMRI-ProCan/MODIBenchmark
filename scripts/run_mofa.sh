# the shell script to run MOFA with different sample sizes

python scripts/run_mofa.py --data-source data --output data/drug/MOFA/mofa_30factor_frac1.0.hdf5 --frac 1 | tee logs/mofa_30factor_frac1.log
python scripts/run_mofa.py --data-source data --output data/drug/MOFA/mofa_30factor_frac0.8.hdf5 --frac 0.8 | tee logs/mofa_30factor_frac0.8.log
python scripts/run_mofa.py --data-source data --output data/drug/MOFA/mofa_30factor_frac0.6.hdf5 --frac 0.6 | tee logs/mofa_30factor_frac0.6.log
python scripts/run_mofa.py --data-source data --output data/drug/MOFA/mofa_30factor_frac0.4.hdf5 --frac 0.4 | tee logs/mofa_30factor_frac0.4.log
python scripts/run_mofa.py --data-source data --output data/drug/MOFA/mofa_30factor_frac0.2.hdf5 --frac 0.2 | tee logs/mofa_30factor_frac0.2.log
