
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
    parser.add_argument('--save-jets',
                        help='save gen jets data in place of gen particles',
                        type=bool,
                        default=False
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

        if not args.save_jets:
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

            if 1.6 < abs(particles.loc[0].eta) < 2.8:
                skim_tree.Fill()
        else:
            # get gen jet data
            jets = dict(e   = np.array(tree.genjet_energy),
                        pt  = np.array(tree.genjet_pt),
                        eta = np.array(tree.genjet_eta),
                        phi = np.array(tree.genjet_phi),
                        n   = np.array(tree.genjet_n)
                        )
            jets = pd.DataFrame(jets)
            
            if jets.query('pt > 20 and 1.6 < abs(eta) < 2.8').shape[0] > 0:
                skim_tree.Fill()

    skim_tree.Print()
    skim_tree.AutoSave()
    skim_file.Close()
    infile.Close()
