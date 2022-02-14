#!/bin/bash

jobid=$1
infile=$2
process=$3
output_dir=$4
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt
NAME=hgcalPythonEnv

echo $jobid 
echo $infile
echo $process
echo $output_dir

# setup local environment
tar xzf source.tar.gz
rm source.tar.gz
source /cvmfs/cms.cern.ch/cmsset_default.sh
source $LCG/setup.sh
source $NAME/bin/activate

mkdir data
python -V
#mv ${infile} data/photons_nopu_ntuples.txt
python scripts/matching.py --config config/matching_cfg.yaml --job_id ${jobid} --input_file ${infile} --is_batch
xrdcp -f data/output_${jobid}.pkl ${output_dir}/output_${process}_${jobid}.pkl

status=$?
exit $status
