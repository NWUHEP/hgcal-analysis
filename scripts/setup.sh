#!/usr/bin/env bash
NAME=hgcalPythonEnv
LCG=/cvmfs/sft.cern.ch/lcg/views/LCG_98python3/x86_64-centos7-gcc9-opt

source $LCG/setup.sh
# following https://aarongorka.com/blog/portable-virtualenv/, an alternative is https://github.com/pantsbuild/pex
python -m venv --copies $NAME
source $NAME/bin/activate
LOCALPATH=$NAME$(python -c 'import sys; print(f"/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")')
export PYTHONPATH=${LOCALPATH}:$PYTHONPATH
python -m pip install setuptools pip --upgrade
python -m pip install uproot --upgrade
python -m pip install awkward --upgrade
python -m pip install pyjet
python -m pip install cython --upgrade
python -m pip install h5py --upgrade
python -m pip install tables --upgrade
python -m pip install h5py --upgrade

#python -m pip install scikit-learn
#python -m pip install xgboost

sed -i '40s/.*/VIRTUAL_ENV="$(cd "$(dirname "$(dirname "${BASH_SOURCE[0]}" )")" \&\& pwd)"/' $NAME/bin/activate
sed -i '1s/#!.*python$/#!\/usr\/bin\/env python/' $NAME/bin/*
sed -i "2a source ${LCG}/setup.sh" $NAME/bin/activate
sed -i "3a export PYTHONPATH=${LOCALPATH}:\$PYTHONPATH" $NAME/bin/activate
sed -i "4a source ${LCG}/setup.csh" $NAME/bin/activate.csh

#tar -zcf env${NAME}.tar.gz ${NAME}
