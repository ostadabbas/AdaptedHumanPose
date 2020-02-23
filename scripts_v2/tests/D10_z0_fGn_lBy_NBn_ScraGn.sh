#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=D10_z0_fGn_lBy_NBn_ScraGn
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:k40m:1	 # resource
#SBATCH --mem=120Gb
#SBATCH --output=bch_outs/D10_z0_fGn_lBy_NBn_ScraGn.%j.out
#SBATCH --error=bch_outs/D10_z0_fGn_lBy_NBn_ScraGn.%j.err
source activate py36
python ${1}.py \
--lmd_D 10. \
--epoch_regZ 0 \
--if_fixG n \
--if_ylB y \
--if_scraG n \
--trainset ScanAva MSCOCO MPII \
--testset SURREAL \
--optimizer adam \
--if_normBone n \
--if_test_ckpt y \
--start_epoch 25 \
--bone_type h36m \
--if_loadPreds n \
--if_aveBoneRec y \
--h36mProto 2 \
--svVis_step 1 \
--test_par train \
--batch_size 30 \
--rt_pelvisUp 0.
