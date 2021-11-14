# 

Tools for analyses related to the development of the HGCal concentrator ASIC.  

### Setup

This code is intended to run independently of CMSSW and only depend on Python.
To this end, we use [miniconda](https://docs.conda.io/en/latest/miniconda.html)
for our base environment and then define a conda environment that will
fulfill the dependencies required to run the code.  The setup should be done in
whatever local computing environment and then the relevant files need to be
placed in an area where they can be transferred to the batch system.  First,
download condor to your local workspace

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Answer `yes` when prompted and then provide your preferred installation
location (the installation with needed dependencies will take up about 3 GB
of disk space).  Create the conda environment and install the needed dependencies:

```
conda create env -n hgcal_econ numpy scipy pandas uproot4=4.1.7 awkward xrootd pytables
```
