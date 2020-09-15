#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=24:00:00
#SBATCH -J mainJobSubmitter
#SBATCH -p short

# change exp names to experiment needs, each run a python train/test codes
#change paths to your desired locations:
SOURCEPATH=scripts  # call this from a sub folder of main
#CURRDIR=/scratch/username/...
N=3  # pay attention to check recursive numbers  D0 2 no need to ran
declare -a arr_exp=(
"D0-SA_yln"
#"D0-SA_yly"
#"D002-C-pn_yln"
"D002-C-py_yln"
#"D002-SA-pn_yln"
"D002-SA-py_yln"
#"D002-SA-py_yly"
                )
# call train ScanAva lsgan
testset=Human36M    # or MuPoTs

for i in "${!arr_exp[@]}"; do
    # single
    JOBID=1
    JOBID=`sbatch --exclude=d1005,d1010,d1020 ${SOURCEPATH}/${arr_exp[$i]}.sh ${1} ${2} ${3} ${testset} | sed 's/>//g;s/<//g' | awk '{print $4}'`
    echo main sbatch job $i ${SOURCEPATH}/${arr_exp[$i]}.sh ${1} ${2} ${3} ${testset} with ID $JOBID at `date`
    # multi
#    for j in $(seq 1 $N)
#    do
#    # multi
##    if [ "$j" -eq "1" ]; then
##    ##retrieve the job id number after submitting the created job script:
###    JOBID=`sbatch --job-name=$i-$j ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`  # with $i-$j job name
##    JOBID=`sbatch ${SOURCEPATH}/${arr_exp[$i]}.sh ${1} | sed 's/>//g;s/<//g' | awk '{print $4}'`  # we can easily see it by same name + jobID
##    else
##    ## if not the first job, submit this job as a dependent of the previous submitted job:
###    JOBID=`sbatch --job-name=$i-$j --dependency=afterok:${JOBID} ${SOURCEPATH}/${arr_exp[$i]}.sh | sed 's/>//g;s/<//g' | awk '{print $4}'`
##    JOBID=`sbatch --dependency=afterok:${JOBID} ${SOURCEPATH}/${arr_exp[$i]}.sh ${1} | sed 's/>//g;s/<//g' | awk '{print $4}'`
##    fi
##    echo main sbatch job $i ${SOURCEPATH}/${arr_exp[$i]}.sh $1 sub $j with ID $JOBID at `date`
#    ##sleep for 1 second to let scheduler update job status properly before submitting more jobs:
#    sleep 1
#    done
    sleep 1
done