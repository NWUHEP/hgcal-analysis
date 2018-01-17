#!/bin/bash
echo "Job submitted on host `hostname` on `date`"
xrdcp root://cmseos.fnal.gov//store/user/jbueghly/hgcal/ntuple_singleGamma_1000.root .
xrdcp root://cmseos.fnal.gov//store/user/jbueghly/hgcal/ntuple_doublenu_1000.root .
xrdcp root://cmseos.fnal.gov//store/user/jbueghly/hgcal/evts.pkl .
xrdcp root://cmseos.fnal.gov//store/user/jbueghly/hgcal/geom.pkl .
tar -xzf python.tar.gz
export PATH=anaconda3/envs/py34root/bin:$PATH
export ROOTSYS=anaconda3/envs/py34root
export PYTHONDIR=anaconda3/envs/py34root
export PATH=$ROOTSYS/bin:$PYTHONDIR/bin:$PATH
export LD_LIBRARY_PATH=$ROOTSYS/lib:$PYTHONDIR/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ROOTSYS/lib:$PYTHONPATH
python test_algorithm.py $1
rm ntuple_singleGamma_1000.root
rm ntuple_doublenu_1000.root
rm evts.pkl
rm geom.pkl
#mkdir home
#export HOME=$(pwd)/home
