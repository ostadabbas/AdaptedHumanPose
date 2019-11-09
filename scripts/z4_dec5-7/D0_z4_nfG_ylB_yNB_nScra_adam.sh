#!/bin/bash
#$SBATCH --job-name=D0_z4_nfG_ylB_yNB_nScra_adam
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/D0_z4_nfG_ylB_yNB_nScra_adam.%j.out
#SBATCH --error=bch_outs/D0_z4_nfG_ylB_yNB_nScra_adam.%j.err
# this short test will feature z4_dec5-7  dec (5, 7 )
source activate py36
python train.py \
--lmd_D 0.1 \
--epoch_regZ 4 \
--if_fixG n \
--if_ylB y \
--if_scraG n \
--trainset ScanAva MSCOCO \
--testset Human36M \
--optimizer adam
