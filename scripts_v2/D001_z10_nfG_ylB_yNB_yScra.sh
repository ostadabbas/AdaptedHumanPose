#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=D001_z10_nfG_ylB_yNB_nScra
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=120Gb
#SBATCH --output=bch_outs/D001_z10_nfG_ylB_yNB_nScra.%j.out
#SBATCH --error=bch_outs/D001_z10_nfG_ylB_yNB_nScra.%j.err
source activate py36
python ${1}.py \
--lmd_D 0.01 \
--epoch_regZ 10 \
--if_fixG n \
--if_ylB y \
--if_scraG y \
--trainset ScanAva MSCOCO MPII \
--testset Human36M \
--optimizer adam
