#!/bin/bash

jobid=$1
infile=$2
echo $1
echo $2

tar xzf source.tar.gz
ll
source $LCG/setup.sh
NAME=hgcalPythonEnv
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt
source $LCG/setup.sh
source $NAME/bin/activate

python scripts/matching.py config/matching_cfg.py --job_id $jobid --input_file $infile

status=$?
exit $status
