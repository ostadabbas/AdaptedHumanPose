#!/bin/bash
#SBATCH --job-name=sa_aaic
#SBATCH --mem=30Gb
#SBATCH --output=sa_aaic.%j.out
#SBATCH --error=sa_aaic.%j.err

python3 sne_plot.py -f /scratch/liu.shu/codesPool/taskGen3d/output/ScanAva_res50_n-scraG_0.0D2_y-yl_1rtSYN_regZ0_n-fG_n-nmBone_adam_lr0.001_exp/vis/train -o SA_AAIC_500.png -m 500 

