#!/bin/bash
#SBATCH --job-name=D001_z0_nfG_ylB_yNB_nScra_nNB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/D001_z0_nfG_ylB_yNB_nScra_nNB.%j.out
#SBATCH --error=bch_outs/D001_z0_nfG_ylB_yNB_nScra_nNB.%j.err
source activate py36
python train.py \
--lmd_D 0.01 \
--epoch_regZ 0 \
--if_fixG n \
--if_ylB y \
--if_scraG n \
--trainset ScanAva MSCOCO MPII \
--testset Human36M \
--optimizer adam \
--if_normBone n