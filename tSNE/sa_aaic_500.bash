#!/bin/bash
#SBATCH --job-name=SA_A
#SBATCH --cpus-per-task=3
#SBATCH --mem=30Gb
#SBATCH --output=SA_A.%j.out
#SBATCH --error=SA_A.%j.err

python3 ex_tsne.py \
-f /scratch/liu.shu/codesPool/AHuP/output/ScanAva-MSCOCO-MPII_res50_n-scraG_0.1D2_y-yl_1rtSYN_regZ10_n-fG_y-nmBone_adam_lr0.001_exp/vis/train \
-m 200 \
--exp sa_aaic


