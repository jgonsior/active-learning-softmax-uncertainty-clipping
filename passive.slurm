#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --time=123:59:59   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=8  # number of processor cores (i.e. threads)
#SBATCH -A p_ml_il
#SBATCH --gres=gpu:1
#SBATCH --mail-user=julius.gonsior@tu-dresden.de
#SBATCH --mem=120GB
#SBATCH --output=/beegfs/ws/1/s5968580-btw/logs/out-%A_%a.txt
#SBATCH --error=/beegfs/ws/1/s5968580-btw/logs/error-%A_%a.txt
#SBATCH --array 0-20


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE=""
module load release/23.04  GCC/11.3.0  OpenMPI/4.1.4
#module load modenv/hiera  GCC/10.2.0  CUDA/11.1.1  OpenMPI/4.0.5
module load PyTorch/1.12.1-CUDA-11.7.0
#module load PyTorch/1.10.0

#source /scratch/ws/1/s5968580-btw/venv/bin/activate
source /beegfs/ws/1/s5968580-btw/python-environments/btw-v3/bin/activate

export HF_MODULE_CACHE='./hf-cache'
export TRANSFORMERS_CACHE="./hf-cache"
export HF_DATASETS_CACHE="./hf-cache"
mkdir -p $TRANSFORMERS_CACHE

#module load PyTorch/1.10.0

#pip install dill scikit-learn tqdm matplotlib seaborn
#pip install torchtext transformers datasets
python /beegfs/ws/1/s5968580-btw/active-learning-softmax-uncertainty-clipping/run_experiment.py --taurus --workload passive --n_array_jobs 20 --array_job_id $SLURM_ARRAY_TASK_ID

exit 0
