#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem 50000M
#SBATCH -t 0-24:00 # time (D-HH:MM)
#SBATCH --job-name="train_multi"
#SBATCH -o slurm/out%j.txt
#SBATCH -e slurm/err%j.txt
#SBATCH --gres=gpu:1
#SBATCH --mail-user=f20210190@hyderabad.bits-pilani.ac.in
#SBATCH --mail-type=ALL
nvidia-smi
conda env list
source activate newtorch
spack load cuda/gypzm3r
spack load cudnn

WORKINGDIR=/home/abhijitdas/multimodalm/TimeSformer/
CURPYTHON=~/anaconda3/envs/mus3ard/bin/python


srun --label ${CURPYTHON} ${WORKINGDIR}/tools/run_net.py --cfg ${WORKINGDIR}/configs/Kinetics/TimeSformer_divST_8x32_224.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8

