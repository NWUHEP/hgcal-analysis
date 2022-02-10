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

tar xzf source.tar.gz
rm source.tar.gz
source $LCG/setup.sh
source $NAME/bin/activate

mkdir data
ls -lh
python scripts/matching.py --help
#python scripts/matching.py --config config/matching_cfg.yaml 
#python scripts/matching.py --config config/matching_cfg.yaml --job_id ${jobid} --input_file ${infile} --output_dir data
#xrdcp -f data/output_${jobid}.hdf5 ${output_dir}/output_${process}_${jobid}.hdf5

status=$?
exit $status
