#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -J mainJobSubmitter
#SBATCH -p general

# change exp names to experiment needs, each run a python train/test codes
#change paths to your desired locations:
SOURCEPATH=scripts_v2  # call this from a sub folder of main
#CURRDIR=/scratch/username/...
N=3  # pay attention to check recursive numbers
declare -a arr_exp=(
#"D0_z0_fGn_lBy_NBn_ScraGn"
#"D0_z0_fGn_lBy_NBn_ScraGy"
#"D01_z0_fGn_lBy_NBn_ScraGn"
##"D001_z0_fGn_lBy_NBn_ScraGn"
#"D1_z0_fGn_lBy_NBn_ScraGn"
#"D10_z0_fGn_lBy_NBn_ScraGn"
#"D100_z0_fGn_lBy_NBn_ScraGn"
#"D100_z0_fGn_lBy_NBn_ScraGy"
#"D100_z0_fGn_lBy_NBn_ScraGn_GmodeV"
#"D1000_z0_fGn_lBy_NBn_ScraGn"
#"SR_D0_z0_fGn_lBy_NBn_ScraGn"
#"SR_D1_z0_fGn_lBy_NBn_ScraGn"
#"SR_D10_z0_fGn_lBy_NBn_ScraGn"
#"SURREAL_ds1"
#"SR_D10_z0_fGn_lBn_NBn_ScraGn"
"MuCo_D0_z0_fGn_lBy_NBn_ScraGn"
#"ScanAva_ds1"
#"D10_z0_fGn_lBn_NBn_ScraGn"
#"D10_z0_fGn_lBy_NBn_ScraGn_rtSyn2"
#"D10_z0_fGn_lBy_NBn_ScraGn_rtSyn3"
#"DS4_D10"
#"D10_z0_fGn_lBy_NBn_ScraGn_res101"
#"D10_z0_fGn_lBy_NBn_ScraGn_res152"
                )

for i in "${!arr_exp[@]}"; do
    for j in $(seq 1 $N)
    do
    ##define some job name
#    jobname=$i.job
    ##replace pattern "JOBNAME" in template script to the defined $jobname variable and create a new submit script $CURRDIR/sub.$i.bash:
#    sed "s/JOBNAME/$jobname/g" $SOURCEPATH/sub.template.bash > $CURRDIR/sub.$i.bash
    ##if this is the first job to be submitted, submit without dependancies:

    if [ "$j" -eq "1" ]; then
    ##retrieve the job id number after submitting the created job script:
#    JOBID=`sbatch --job-name=$i-$j ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`  # with $i-$j job name
    JOBID=`sbatch ${SOURCEPATH}/${arr_exp[$i]}.sh ${1} | sed 's/>//g;s/<//g' | awk '{print $4}'`  # we can easily see it by same name + jobID
    else
    ## if not the first job, submit this job as a dependent of the previous submitted job:
#    JOBID=`sbatch --job-name=$i-$j --dependency=afterok:${JOBID} ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`
    JOBID=`sbatch --dependency=afterok:${JOBID} ${SOURCEPATH}/${arr_exp[$i]}.sh ${1} | sed 's/>//g;s/<//g' | awk '{print $4}'`
    fi
    echo main sbatch job $i ${SOURCEPATH}/${arr_exp[$i]}.sh $1 sub $j with ID $JOBID at `date`
    ##sleep for 1 second to let scheduler update job status properly before submitting more jobs:
    sleep 1
    done
done