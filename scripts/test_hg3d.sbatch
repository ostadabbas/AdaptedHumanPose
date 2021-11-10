#!/bin/bash
#SBATCH --job-name=taskGen3d
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:v100-sxm2:1	 # resource
#SBATCH --mem=128Gb
#SBATCH --time=24:00:00
#SBATCH --output=bch_outs/taskGen3d.%j.out
#SBATCH --error=bch_outs/taskGen3d.%j.err
# model needs to be changed to  hg3d (based on p2) as expNm , test the pretraind performance


source activate pch1.5
python test_hg3d.py \
--model evoSkel \
--lmd_D 0.02 \
--mode_D SA \
--pivot sdt \
--gan_mode lsgan \
--if_ylB y \
--testset ${1} \
--h36mProto 2 \
--test_par test \
--testIter -1 \
--if_loadPreds ${2:-n} \
--if_hm_PA ${3:-y}
