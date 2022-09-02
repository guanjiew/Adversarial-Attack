#!/bin/bash
#SBATCH -p rtx6000,p100,t4v2
#SBATCH --qos=normal
#SBATCH -x gpu095,gpu111,gpu115
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --job-name=train
#SBATCH --array=0-44%45
#SBATCH --output=output/task_%A_%a.log
#SBATCH --open-mode=append

date
hostname
pwd
source $HOME/.bashrc
conda activate al
cd $HOME/Project/adversarial_attack

echo "This is SLURM task $SLURM_ARRAY_TASK_ID"

PATH_CONFIG="--experiment_name=exp_4_15 --resume=True --dataset=emnist --depth=16 --epochs=150 --lr=0.03 --schedule 75 110"
CONFIG_A="--widen-factor=1"
CONFIG_B="--widen-factor=2"
CONFIG_C="--widen-factor=4"
CONFIG_D="--widen-factor=8"
CONFIG_E="--widen-factor=12"
list=(
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=FGSM --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=FGSM --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=FGSM --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=PGD --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=PGD --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=PGD --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=FAB --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=FAB --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_A --attack_method=FAB --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=FGSM --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=FGSM --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=FGSM --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=PGD --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=PGD --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=PGD --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=FAB --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=FAB --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_B --attack_method=FAB --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=FGSM --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=FGSM --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=FGSM --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=PGD --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=PGD --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=PGD --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=FAB --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=FAB --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_C --attack_method=FAB --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=FGSM --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=FGSM --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=FGSM --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=PGD --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=PGD --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=PGD --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=FAB --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=FAB --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_D --attack_method=FAB --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=FGSM --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=FGSM --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=FGSM --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=PGD --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=PGD --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=PGD --manualSeed=2"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=FAB --manualSeed=0"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=FAB --manualSeed=1"
  "python main2.py $PATH_CONFIG $CONFIG_E --attack_method=FAB --manualSeed=2"

)

${list[SLURM_ARRAY_TASK_ID]}
