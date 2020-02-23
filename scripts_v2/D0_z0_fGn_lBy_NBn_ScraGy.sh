#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=D0_z0_fGn_lBy_NBn_ScraGy
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=120Gb
#SBATCH --output=bch_outs/D0_z0_fGn_lBy_NBn_ScraGy.%j.out
#SBATCH --error=bch_outs/D0_z0_fGn_lBy_NBn_ScraGy.%j.err
source activate py36
python ${1}.py \
--lmd_D 0. \
--epoch_regZ 0 \
--if_fixG n \
--if_ylB y \
--if_scraG y \
--trainset ScanAva MSCOCO MPII \
--testset Human36M \
--optimizer adam \
--if_normBone n
