#!/bin/bash
#SBATCH --partition=gpu2,ml,alpha
#SBATCH --time=8:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -A p_ml_il
#SBATCH --gres=gpu:1
#SBATCH --mail-user=julius.gonsior@tu-dresden.de
#SBATCH --mem=120GB
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output /scratch/ws/1/s5968580-btw/out.txt
#SBATCH --error /scratch/ws/1/s5968580-btw/error.txt


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE=""

module load Python/3.10.4-GCCcore-11.3.0-bare
source /scratch/ws/1/s5968580-btw/venv/bin/activate
python /scratch/ws/1/s5968580-btw/code/test.py --num_iterations 5 --batch_size 20 --exp_name test_taurus --dataset trec6 --random_seed 2000 --query_strategy LC
#/scratch/ws/1/s5968580-btw/venv/bin/python /scratch/ws/1/s5968580-btw/code/test.py --num_iterations 5 --batch_size 20 --exp_name test_taurus --dataset trec6 --random_seed 2000 --query_strategy LC


exit 0
