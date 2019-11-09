#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/z10_nfG_ylB_yNB.%j.out
#SBATCH --error=bch_outs/z10_nfG_ylB_yNB.%j.err
# very safe version, only working on real with scraG with adams to see 2d effect
source activate py36
python train.py \
--lmd_D 0 \
--epoch_regZ 4 \
--if_fixG n \
--if_ylB y \
--if_scraG n \
--trainset MPII MSCOCO \
--testset Human36M \
--optimizer adam
