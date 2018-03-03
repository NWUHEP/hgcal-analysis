import argparse
import pickle
from collections import namedtuple
import pandas as pd
import numpy as np

import ROOT as r

# container for gen particle data
from tqdm import trange
particle = namedtuple('particle', ['e', 'pt', 'eta', 'phi', 'ex', 'ey'])

def unpack_tree(tree, evt, ievt=1, is_pileup=False):
    tree.GetEntry(evt)

    # make trigger cell dataframe
    df_tmp = dict(ievt    = np.array(tree.tc_n*[int(evt), ], dtype=int),
                  x       = np.array(tree.tc_x),
                  y       = np.array(tree.tc_y),
                  zside   = np.array(tree.tc_zside),
                  layer   = np.array(tree.tc_layer, dtype=int),
                  subdet  = np.array(tree.tc_subdet, dtype=int),
                  panel   = np.array(tree.tc_panel_number, dtype=int),
                  sector  = np.array(tree.tc_panel_sector, dtype=int),
                  cell    = np.array(tree.tc_cell, dtype=int),
                  reco_e  = np.array(tree.tc_energy),
                  sim_e   = np.array(tree.tc_simenergy) if not is_pileup else np.zeros(tree.tc_n),
                  )
    df_tmp = pd.DataFrame(df_tmp)
    return df_tmp

def get_genpart(tree, evt):
    tree.GetEntry(evt)

    # save gen particle data
    particles = particle(e   = np.array(tree.genpart_energy),
                         eta = np.array(tree.genpart_eta),
                         phi = np.array(tree.genpart_phi),
                         ex  = np.array(tree.genpart_exx),
                         pt  = np.array(tree.genpart_pt),
                         ey  = np.array(tree.genpart_exy)
                        )

    return particles


if __name__ == '__main__':

    # parse arguments #
    parser = argparse.ArgumentParser(description='Convert hgcal root ntuples to dataframes')
    parser.add_argument('signal_input',
                        help='root file containing signal process',
                        type=str
                        )
    parser.add_argument('pileup_input',
                        help='root file containing pure pileup samples',
                        type=str
                        )
    parser.add_argument('-n', '--nepochs',
                        help='number of epochs',
                        type=int
                        )
    parser.add_argument('-b', '--bunch-pattern',
                        help='specifies bunch pattern',
                        type=str
                        )
    args = parser.parse_args()
    ###############################

    # Get input files
    signal_filename = args.signal_input
    pileup_filename = args.pileup_input
    output_filename = signal_filename.split('/')[-1].split('.')[0]

    signal_file = r.TFile(signal_filename)
    signal_tree = signal_file.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    n_signal    = signal_tree.GetEntriesFast()
    signal_evts = np.arange(n_signal)

    pileup_file = r.TFile(pileup_filename)
    pileup_tree = pileup_file.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    n_pileup    = pileup_tree.GetEntriesFast()
    pileup_evts = np.arange(n_pileup)

    if isinstance(args.nepochs, type(None)):
        n_epochs = 10
    else:
        n_epochs = args.nepochs

    bx = np.arange(8)
    df_list  = []
    gen_list = []
    mi_labels = ['zside', 'layer', 'sector', 'panel', 'cell']
    for i in trange(n_epochs):
        np.random.shuffle(bx)
        isig  = np.random.choice(signal_evts, 1)
        ibg   = np.random.choice(pileup_evts, 7)

        # get gen particle collection
        gen_particles = [get_genpart(signal_tree, n) for n in isig]
        gen_list.append(gen_particles)

        # get signal and pileup data and concatenate into a dataframe
        signal_list = [unpack_tree(signal_tree, n, ievt=bx[j]) for j, n in enumerate(isig)]
        pileup_list = [unpack_tree(pileup_tree, n, ievt=bx[j+1], is_pileup=True) for j, n in enumerate(ibg)]
        df = pd.concat(signal_list + pileup_list)

        # if there are no simhits in hgcal, we don't care about this event
        if df.sim_e.sum() == 0.:
            continue

        # only save data from panels that have simhits (not great...)
        #df_skim = df.query('sim_e > 0.')[mi_labels]
        #df_skim = df_skim.drop_duplicates().reset_index(drop=True)
        #gr_tmp  = df.groupby(['zside', 'layer', 'sector', 'panel'])
        #df_tmp  = [gr_tmp.get_group(tuple(r)) for r in df_skim.values[:, :-1]]
        #df      = pd.concat(df_tmp)

        df_list.append(df)

    output_file = open(f'data/mc_mixtures/{output_filename}_{n_epochs}.pkl', 'wb')
    pickle.dump(gen_list, output_file)
    pickle.dump(df_list, output_file)

    signal_file.Close()
    pileup_file.Close()
    output_file.close()
