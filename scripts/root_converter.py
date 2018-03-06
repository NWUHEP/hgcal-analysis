
import argparse
import pickle
from functools import partial
import pandas as pd
import numpy as np
import ROOT as r


# container for gen particle data
from tqdm import trange

def unpack_tree(tree, is_pileup=False):

    # make trigger cell dataframe
    df_tmp = dict(x       = np.array(tree.tc_x),
                  y       = np.array(tree.tc_y),
                  zside   = np.array(tree.tc_zside),
                  layer   = np.array(tree.tc_layer, dtype=int),
                  subdet  = np.array(tree.tc_subdet, dtype=int),
                  sector  = np.array(tree.tc_panel_sector, dtype=int),
                  panel   = np.array(tree.tc_panel_number, dtype=int),
                  wafer   = np.array(tree.tc_wafer, dtype=int),
                  cell    = np.array(tree.tc_cell, dtype=int),
                  pt      = np.array(tree.tc_pt),
                  mip_pt  = np.array(tree.tc_mipPt),
                  reco_e  = np.array(tree.tc_energy),
                  sim_e   = np.array(tree.tc_simenergy) if not is_pileup else np.zeros(tree.tc_n),
                  )
    df_tmp = pd.DataFrame(df_tmp)

    return df_tmp

def get_genpart(tree):
    # save gen particle data
    particles = dict(e   = np.array(tree.genpart_energy),
                     eta = np.array(tree.genpart_eta),
                     phi = np.array(tree.genpart_phi),
                     ex  = np.array(tree.genpart_exx),
                     pt  = np.array(tree.genpart_pt),
                     ey  = np.array(tree.genpart_exy),
                     pid = np.array(tree.genpart_pid)
                    )
    particles = pd.DataFrame(particles)
    #particles = particles[:2]
    return particles

def threshold_algos(df, sort_by, name, ascending=False, nbx=8):
    df = df.sort_values(sort_by, ascending=ascending)
    df.loc[:,name] = True
    if df.shape[0] > nbx*10:
        df.loc[nbx*10:,name] = False

    return df

def algos_8bx(df):

    # 8bx no sort
    df = df.reset_index(drop=True) # causes problems otherwise
    threshold_algos(df, 
                    sort_by=['ievt', 'cell'], 
                    name='threshold_8bx_nosort',
                    ascending=True, 
                    nbx=8
                    )
    # 8bx energy sort
    threshold_algos(df, 
                    sort_by='reco_e', 
                    name='threshold_8bx_esort',
                    ascending=False, 
                    nbx=8
                    )
    return df

def algos_1bx(df):

    # 8bx no sort
    df = df.reset_index(drop=True) # causes problems otherwise
    threshold_algos(df, 
                    sort_by=['ievt', 'cell'], 
                    name='threshold_1bx_nosort',
                    ascending=True, 
                    nbx=1
                    )
    # 8bx energy sort
    threshold_algos(df, 
                    sort_by='reco_e', 
                    name='threshold_1bx_esort',
                    ascending=False, 
                    nbx=1
                    )
    return df

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
                        help='specifies bunch pattern (not currently implented)',
                        type=str
                        )
    args = parser.parse_args()
    ###############################

    # Get input files
    signal_filename = args.signal_input
    pileup_filename = args.pileup_input
    output_filename = signal_filename.split('/')[-1].split('.')[0]

    if isinstance(args.nepochs, type(None)):
        n_epochs = 10
    else:
        n_epochs = args.nepochs

    pileup_file = r.TFile(pileup_filename)
    pileup_tree = pileup_file.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    n_pileup = pileup_tree.GetEntriesFast()
    pileup_dfs = []
    for i in trange(7*n_epochs, desc='Getting pileup events'):
        pileup_tree.GetEntry(i%n_pileup)

        # make trigger cell dataframe
        df_tmp = unpack_tree(pileup_tree, is_pileup=True)
        pileup_dfs.append(df_tmp)

    signal_file = r.TFile(signal_filename)
    signal_tree = signal_file.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    n_signal = signal_tree.GetEntriesFast()
    signal_dfs = []
    gen_list = []
    for i in trange(2*n_epochs, desc='Getting signal events'):
        signal_tree.GetEntry(i%n_signal)

        # make trigger cell dataframe
        df_tmp = unpack_tree(signal_tree)
        if df_tmp.sim_e.sum() == 0:
            continue

        ## get gen particle collection
        gen_particles = get_genpart(signal_tree)
        if 1.7 < abs(gen_particles.loc[0].eta) < 2.7:
            continue

        signal_dfs.append(df_tmp)
        gen_list.append(gen_particles)

    # some useful data
    pileup_evts = np.arange(len(signal_dfs))
    signal_evts = np.arange(len(signal_dfs))
    bx          = list(range(8))
    mi_labels   = ['zside', 'layer', 'sector', 'panel']

    # make df_lists and test algorithms
    df_list    = []
    for i in trange(n_epochs):
        np.random.shuffle(bx)
        isig = list(np.random.choice(signal_evts, 1))
        ibg  = list(np.random.choice(pileup_evts, 7))
        
        df_sig = signal_dfs[isig[0]]
        df_sig['ievt'] = bx[0]

        bg_list = []
        for cnt, j in enumerate(ibg):
            df_bg = pileup_dfs[j]
            df_bg['ievt'] = bx[cnt + 1]
            bg_list.append(df_bg)

        df = pd.concat(bg_list + [df_sig])

        # only save data from panels that have simhits 
        df = df.groupby(mi_labels).filter(lambda x: x.sim_e.sum() > 0)

        # carry out the readout algorithms here
        df = df.groupby(mi_labels, sort=False).apply(algos_8bx)
        df = df.groupby(mi_labels+['ievt'], sort=False).apply(algos_1bx)

        # save dataframes for making plots
        df_list.append(df)

    # just for debugging
    #df = df_list[-1]

    file_count = 0
    output_file = open(f'data/mc_mixtures/{output_filename}_{n_epochs}_{file_count}.pkl', 'wb')
    pickle.dump(gen_list, output_file)
    pickle.dump(df_list, output_file)
    output_file.close()

    signal_file.Close()
    pileup_file.Close()
