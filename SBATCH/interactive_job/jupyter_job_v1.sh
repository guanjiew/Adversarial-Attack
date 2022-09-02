#!/bin/bash
#SBATCH -p interactive
#SBATCH --gres=gpu:1
#SBATCH --qos=nopreemption
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --job-name=jupyter
#SBATCH --output=output/jupyter_notebook_%j.log
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

date;hostname;pwd
echo "cat output/jupyter_$SLURM_ARRAY_TASK_ID.log"
source $HOME/.bashrc
conda activate al
cd $HOME/adversarial_attack
. al.env
export XDG_RUNTIME_DIR=""
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/pkgs/cuda-10.0
jupyter notebook --ip 0.0.0.0 --port 6699