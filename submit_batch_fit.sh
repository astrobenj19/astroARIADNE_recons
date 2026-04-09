#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=normal
#SBATCH --nodelist=cn[001-002]


export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48

. /home/way/.bashrc

python --version

#python ariadne_batch_fit.py 
#python ariadne_batch_fit.py --csv-path photometry.csv --nlive 50 --n-samples 1000 --dlogz 0.999 --threads 48 --plot-sed  --plot-corner --num-targets 5
python ariadne_batch_fit.py --threads 48 --csv-path photometry.csv --nlive 500 --n-samples 1000 --dlogz 0.100 --plot-all --num-targets 5
