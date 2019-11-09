#!/bin/bash
#SBATCH --job-name=D0_R2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/D0_R2.%j.out
#SBATCH --error=bch_outs/D0_R2.%j.err
#SBATCH --exclusive
# very safe version, only working on real with scraG with adams to see 2d effect
source activate py36
python train.py \
--lmd_D 0 \
--epoch_regZ 0 \
--if_fixG n \
--if_ylB y \
--if_scraG n \
--trainset MPII MSCOCO \
--testset Human36M \
--optimizer adam
