import argparse
import pickle
import pandas as pd
import numpy as np

from tqdm import tqdm, trange
import ROOT as r

def unpack_tree(tree, evt, is_pileup=False):
    tree.GetEntry(evt)
    df_tmp = dict(
                  ievt    = np.array(tree.tc_n*[int(evt),], dtype=int),
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
    #df_tmp['delta_e'] = df_tmp['sim_e'] - df_tmp['reco_e'],
    df_tmp = pd.DataFrame(df_tmp)
    return df_tmp


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

    if type(args.nepochs) == type(None):
        n_epochs = 10
    else:
        n_epochs = args.nepochs

    df_list = []
    mi_labels = ['zside', 'layer', 'sector', 'panel', 'cell']
    for i in trange(n_epochs):

        # get signal and pileup data and concatenate into a dataframe
        signal_list = [unpack_tree(signal_tree, n) for n in np.random.choice(signal_evts, 1)]
        pileup_list = [unpack_tree(pileup_tree, n, is_pileup=True) for n in np.random.choice(pileup_evts, 1)]
        df = pd.concat(signal_list + pileup_list)
        
        if df.sim_e.sum() == 0.: continue

        # only save data from panels that have simhits
        df_skim = df.query('sim_e > 0.')[mi_labels]
        df_skim = df_skim.drop_duplicates().reset_index(drop=True)
        panel_indices = [tuple(r) for r  in df_skim.values[:,:-1]]
        df = df.set_index(mi_labels)
        #df = df.loc[panel_indices]
        df_list.append(df)

    output_file = open(f'data/mc_mixtures/{output_filename}_{n_epochs}.pkl', 'wb')
    pickle.dump(df_list, output_file)

    signal_file.Close()
    pileup_file.Close()
    output_file.close()


