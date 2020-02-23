#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=H36M_MPII
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=120Gb
#SBATCH --output=bch_outs/H36M_MPII.%j.out
#SBATCH --error=bch_outs/H36M_MPII.%j.err

# cam dist official h36m, test if my structure is correct as original one
source activate py36
python ${1}.py \
--lmd_D 0 \
--epoch_regZ 0 \
--if_fixG n \
--if_ylB y \
--if_scraG n \
--trainset Human36M MPII \
--testset Human36M \
--optimizer adam \
--epoch_step  3 \
--save_step 1
