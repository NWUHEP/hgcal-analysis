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


def unpack_tree(tree, is_pileup=False):
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
                  #sim_e   = np.array(tree.tc_simenergy) if not is_pileup else np.zeros(tree.tc_n),
                  )
    df_tmp = pd.DataFrame(df_tmp)

    return df_tmp

def get_genpart(tree):
    # save gen particle data
    particles = dict(e   = np.array(tree.genpart_energy),
                     eta = np.array(tree.genpart_eta),
                     phi = np.array(tree.genpart_phi),
                     pt  = np.array(tree.genpart_pt),
                     dvx = np.array(tree.genpart_dvx),
                     dvy = np.array(tree.genpart_dvy),
                     dvz = np.array(tree.genpart_dvz),
                     ovx = np.array(tree.genpart_ovx),
                     ovy = np.array(tree.genpart_ovy),
                     ovz = np.array(tree.genpart_ovz),
                     ex  = np.array(tree.genpart_exx),
                     ey  = np.array(tree.genpart_exy),
                     pid = np.array(tree.genpart_pid)
                    )
    particles = pd.DataFrame(particles)
    #particles = particles[:2]
    return particles


def convert_tree(run_data):

    file_type   = run_data['file_type']
    file_count  = run_data['file_count']
    output_dir  = run_data['output_dir']

    # some useful data
    rootfile = r.TFile(run_data['filepath'])
    #if (file_type == 'pileup'):
    #    tree = rootfile.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple') # this depends on the file
    #else:
    #    tree = rootfile.Get('HGCalTriggerNtuple') # this depends on the file

    tree = rootfile.Get('HGCalTriggerNtuple') # this depends on the file
    n_events = tree.GetEntriesFast()

    data_list = []
    gen_list  = []
    for i in trange(n_events):
        
        # get signal data
        tree.GetEntry(i)
        df = unpack_tree(tree, is_pileup=(file_type == 'pileup'))
        data_list.append(df)

        gen_df = get_genpart(tree)
        gen_list.append(gen_df)

        if i%1000 == 0 and i > 0:
            output_file = open(f'{output_dir}/output_{file_count}.pkl', 'wb')
            pickle.dump(gen_list, output_file)
            pickle.dump(data_list, output_file)
            output_file.close()
            file_count += 1

            gen_list  = []
            data_list = []

    if len(data_list) > 0:
        output_file = open(f'{output_dir}/output_{file_count}.pkl', 'wb')
        pickle.dump(gen_list, output_file)
        pickle.dump(data_list, output_file)
        output_file.close()

    rootfile.Close()

    return True 

if __name__ == '__main__':

    # parse arguments #
    parser = argparse.ArgumentParser(description='Convert hgcal root ntuples to dataframes')
    parser.add_argument('input',
                        help='input root file',
                        type=str
                        )
    parser.add_argument('-p', '--processes',
                        help='number of concurrent processes to run',
                        default=1,
                        type=int
                        )
    parser.add_argument('-t', '--file-type',
                        help='file type',
                        default='normal',
                        type=str
                        )
    args = parser.parse_args()
    ###############################

    # unpack arguments
    filepath   = args.input
    n_process  = args.processes
    input_name = filepath.split('/')[-1].split('.')[0]

    # multiprocess the data mixing and algorithm testing
    output_dir = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    run_data    = dict(file_count = 0,
                       filepath   = filepath,
                       output_dir = output_dir,
                       file_type  = args.file_type
                       ) 
    convert_tree(run_data)

    #run_data    = [dict(file_count = i,
    #                    filepath   = filepath,
    #                    output_dir = output_dir,
    #                    n_process  = n_process,
    #                    file_type  = args.file_type
    #                    ) for i in range(n_process)]
    #pool        = Pool(processes = n_process)
    #pool_result = pool.map(convert_tree, run_data)

    #pool.close()
    #pool.join()
    #pool.close()
