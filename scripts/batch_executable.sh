#!/bin/bash

jobid=$1
infile=$2
echo $1
echo $2

tar xzf source.tar.gz
#xrdcp -r root://cmseos.fnal.gov//store/user/naodell/miniconda .

# setup python environment w/o CMSSW (how?)
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/uscms/home/naodell/nobackup/opt/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/uscms/home/naodell/nobackup/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/uscms/home/naodell/nobackup/opt/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/uscms/home/naodell/nobackup/opt/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

python matching.py config/matching_cfg.py --job_id $jobid --input_file $infile

status=$?
exit $status

