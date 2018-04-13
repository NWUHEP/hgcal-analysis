
import argparse
import pickle
from functools import partial
import pandas as pd
import numpy as np
import ROOT as r

# container for gen particle data
from tqdm import trange

if __name__ == '__main__':

    # parse arguments #
    parser = argparse.ArgumentParser(description='Convert hgcal root ntuples to dataframes')
    parser.add_argument('input',
                        help='root file to be skimmed',
                        type=str
                       )
    args = parser.parse_args()
    ###############################

    # Get input files
    filename = args.input
    infile = r.TFile(filename)
    tree = infile.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    n_entries = tree.GetEntriesFast()

    skim_file = r.TFile(filename.split('.')[0] + '_skim.root', 'recreate')
    skim_tree = tree.CloneTree(0)
    for i in trange(n_entries):
        tree.GetEntry(i)

        # get gen particle data
        particles = dict(e   = np.array(tree.genpart_energy),
                         pt  = np.array(tree.genpart_pt),
                         eta = np.array(tree.genpart_eta),
                         phi = np.array(tree.genpart_phi),
                         ex  = np.array(tree.genpart_exx),
                         ey  = np.array(tree.genpart_exy),
                         pid = np.array(tree.genpart_pid)
                        )
        particles = pd.DataFrame(particles)
        if particles.shape[0] < 2:
            continue

        if 1.7 < abs(particles.loc[0].eta) < 2.7:
            skim_tree.Fill()

    skim_tree.Print()
    skim_tree.AutoSave()
    skim_file.Close()
    infile.Close()
