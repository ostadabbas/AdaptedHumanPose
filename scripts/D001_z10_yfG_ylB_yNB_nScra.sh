#!/bin/bash
#SBATCH --job-name=D001_z10_yfG_ylB_yNB_nScra
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=100Gb
#SBATCH --output=bch_outs/D001_z10_yfG_ylB_yNB_nScra.%j.out
#SBATCH --error=bch_outs/D001_z10_yfG_ylB_yNB_nScra.%j.err
source activate py36
python train.py \
--lmd_D 0.01 \
--epoch_regZ 10 \
--if_fixG y \
--if_ylB y \
--if_scraG n \
--trainset ScanAva MSCOCO MPII \
--testset Human36M \
--optimizer adam
