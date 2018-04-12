import os
import argparse
import pickle
from functools import partial
from multiprocessing import Process, Pool
import pandas as pd
import numpy as np

import ROOT as r
from tqdm import trange

import scripts.readout_algorithms as algos

def get_current_time():
    import datetime
    now = datetime.datetime.now()
    currentTime = '{0:02d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
    return currentTime


def unpack_tree(tree, algo_list, is_pileup=False):
    # make trigger cell dataframe
    df_tmp = dict(x       = np.array(tree.tc_x),
                  y       = np.array(tree.tc_y),
                  z       = np.array(tree.tc_z),
                  zside   = np.array(tree.tc_zside),
                  layer   = np.array(tree.tc_layer, dtype=int),
                  subdet  = np.array(tree.tc_subdet, dtype=int),
                  sector  = np.array(tree.tc_panel_sector, dtype=int),
                  panel   = np.array(tree.tc_panel_number, dtype=int),
                  wafer   = np.array(tree.tc_wafer, dtype=int),
                  cell    = np.array(tree.tc_cell, dtype=int),
                  eta     = np.array(tree.tc_eta),
                  phi     = np.array(tree.tc_phi),
                  pt      = np.array(tree.tc_pt),
                  mip_pt  = np.array(tree.tc_mipPt),
                  reco_e  = np.array(tree.tc_energy),
                  sim_e   = np.array(tree.tc_simenergy) if not is_pileup else np.zeros(tree.tc_n),
                  )
    df_tmp  = pd.DataFrame(df_tmp)
    df_algo = pd.DataFrame({f'{algo}': np.zeros(df_tmp.shape[0], dtype=bool) for algo in algo_list})

    return pd.concat([df_tmp, df_algo], axis=1)

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

def get_genjets(tree):
    # save gen jeticle data
    jets = dict(e   = np.array(tree.genjet_energy),
                eta = np.array(tree.genjet_eta),
                phi = np.array(tree.genjet_phi),
                pt  = np.array(tree.genjet_pt),
                n   = np.array(tree.genjet_n),
                )
    jets = pd.DataFrame(jets)
    return jets


def mix_and_analyze(run_data):

    file_count      = run_data['file_count']
    n_epochs        = run_data['n_epochs']
    output_dir      = run_data['output_dir']
    mippt_threshold = run_data['mippt_threshold']
    save_jets       = run_data['save_jets']

    # some useful data
    pileup_file = r.TFile(run_data['pileup_filename'])
    pileup_tree = pileup_file.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    n_pileup    = pileup_tree.GetEntriesFast()
    pileup_evts = list(range(n_pileup - 7)) # leave off the last seven events for sampling purposes

    signal_file = r.TFile(run_data['signal_filename'])
    signal_tree = signal_file.Get('HGCalTriggerNtuple')
    n_signal    = signal_tree.GetEntriesFast()
    signal_evts = list(range(n_signal))

    np.random.seed()
    bx = list(range(8))

    # algorithms to test
    algo_list = [
                 'threshold_1bx_esort', 'threshold_8bx_esort', 
                 'threshold_1bx_nosort', 'threshold_8bx_nosort'
                 ]

    # make df_lists and test algorithms
    data_list = []
    gen_list  = []
    mi_labels = ['zside', 'layer', 'sector', 'panel']
    for i in trange(n_epochs, position=file_count, leave=True): #, description=f'process {file_count}'):
        np.random.shuffle(bx)
        isig = np.random.choice(signal_evts)
        ibg  = np.random.choice(pileup_evts)
        
        # get signal data
        signal_tree.GetEntry(isig)
        df_sig = unpack_tree(signal_tree, algo_list)
        df_sig['ievt'] = bx[0]
        df_sig['sig_evt'] = True
        df_sig = df_sig.query(f'mip_pt > {mippt_threshold}')

        # get pileup data
        bg_list = []
        for cnt, j in enumerate(range(7)):
            pileup_tree.GetEntry(ibg+j)
            df_bg = unpack_tree(pileup_tree, algo_list, is_pileup=True)
            df_bg['ievt'] = bx[cnt + 1]
            df_bg['sig_evt'] = False
            df_bg = df_bg.query(f'mip_pt > {mippt_threshold}')
            bg_list.append(df_bg)

        df = pd.concat(bg_list + [df_sig], axis=0)

        # only save data from panels that have simhits 
        df = df.groupby(mi_labels).filter(lambda x: x.sim_e.sum() > 0)
        df = df.reset_index(drop=True) 

        # apply readout algorithms
        df = df.groupby(mi_labels).apply(algos.algorithm_test_8bx)
        df = df.reset_index(drop=True) 
        df = df.groupby(mi_labels + ['ievt']).apply(algos.algorithm_test_1bx)

        # get gen objects
        if save_jets:
            gen_df = get_genjets(signal_tree)
        else:
            gen_df = get_genpart(signal_tree)

        # save dataframes for making plots
        df = df.reset_index(drop=True) # don't save heirarchical indices
        data_list.append(df)
        gen_list.append(gen_df)

    signal_file.Close()
    pileup_file.Close()

    output_file = open(f'{output_dir}/output_{file_count}.pkl', 'wb')
    pickle.dump(mippt_threshold, output_file)
    pickle.dump(gen_list, output_file)
    pickle.dump(data_list, output_file)
    output_file.close()

    return True 

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
                        default=10,
                        type=int
                        )
    parser.add_argument('-p', '--processes',
                        help='number of concurrent processes to run',
                        default=8,
                        type=int
                        )
    parser.add_argument('-b', '--bunch-pattern',
                        help='specifies bunch pattern (not currently implented)',
                        type=str
                        )
    parser.add_argument('--threshold',
                        help='threshold to place on the minimum mip pt considered when simulating readout',
                        default=2.,
                        type=float
                        )
    parser.add_argument('--save-jets',
                        type=bool,
                        default=False,
                        help='switch to saving gen jets instead of individual particles'
                        )
    args = parser.parse_args()
    ###############################

    # unpack arguments
    n_process       = args.processes
    n_epochs        = args.nepochs
    signal_filename = args.signal_input
    pileup_filename = args.pileup_input
    input_name      = signal_filename.split('/')[-1].split('.')[0]

    # multiprocess the data mixing and algorithm testing
    #output_dir = f'data/mc_mixtures/{input_name}_{get_current_time()}'
    output_dir = f'data/mc_mixtures/{input_name}_test'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        os.system(f'rm -r {output_dir}/*')

    #mipscan = np.arange(2, 11, 8/n_process)
    run_data    = [dict(file_count      = i,
                        n_epochs        = n_epochs,
                        signal_filename = signal_filename,
                        pileup_filename = pileup_filename,
                        output_dir      = output_dir,
                        save_jets       = args.save_jets,
                        mippt_threshold = i + 2
                        ) for i in range(n_process)]
    pool        = Pool(processes = n_process)
    pfunc       = partial(mix_and_analyze)
    pool_result = pool.map(pfunc, run_data)

    pool.close()
    pool.join()
    pool.close()
