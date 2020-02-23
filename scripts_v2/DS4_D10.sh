#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=DS4_D10
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=120Gb
#SBATCH --output=bch_outs/DS4_D10.%j.out
#SBATCH --error=bch_outs/DS4_D10.%j.err
source activate py36
python ${1}.py \
--lmd_D 10. \
--epoch_regZ 0 \
--if_fixG n \
--if_ylB y \
--if_scraG n \
--trainset ScanAva Human36M MSCOCO MPII \
--testset Human36M \
--optimizer adam \
--if_normBone n \
--epoch_step 7
