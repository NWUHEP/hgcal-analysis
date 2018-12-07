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


def get_tc(tree, is_pileup=False):
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
                  tc_pt   = np.array(tree.tc_pt),
                  tc_eta  = np.array(tree.tc_eta),
                  tc_phi  = np.array(tree.tc_phi),
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

def get_clusters(tree):
    # save gen particle data
    clusters = dict(
                    cl_pt        = np.array(tree.cl3d_pt),
                    cl_e         = np.array(tree.cl3d_energy),
                    cl_eta       = np.array(tree.cl3d_eta),
                    cl_phi       = np.array(tree.cl3d_phi),
                    n            = np.array(tree.cl3d_n),
                    id           = np.array(tree.cl3d_id),
                    clusters_n   = np.array(tree.cl3d_clusters_n),
                    showerlength = np.array(tree.cl3d_showerlength),
                    corelength   = np.array(tree.cl3d_coreshowerlength),
                    seetot       = np.array(tree.cl3d_seetot),
                    seemax       = np.array(tree.cl3d_seemax),
                    spptot       = np.array(tree.cl3d_spptot),
                    sppmax       = np.array(tree.cl3d_sppmax),
                    srrtot       = np.array(tree.cl3d_srrtot),
                    srrmax       = np.array(tree.cl3d_srrmax),
                    srrmean      = np.array(tree.cl3d_srrmean),
                    szz          = np.array(tree.cl3d_szz),
                   )
    clusters = pd.DataFrame(clusters)
    return clusters


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

    tree = rootfile.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple') # this depends on the file
    n_events = tree.GetEntriesFast()

    tc_list      = []
    gen_list     = []
    cluster_list = []
    for i in trange(n_events):
        
        # get signal data
        tree.GetEntry(i)

        if len(tree.genpart_pid) == 0:
            continue

        df_gen = get_genpart(tree)
        gen_list.append(df_gen)

        df_tc = get_tc(tree, is_pileup=(file_type == 'pileup'))
        tc_list.append(df_tc)

        df_cluster = get_clusters(tree)
        cluster_list.append(df_cluster)

        if i%1000 == 0 and i > 0:
            output_file = open(f'{output_dir}/output_{file_count}.pkl', 'wb')
            pickle.dump(gen_list, output_file)
            pickle.dump(tc_list, output_file)
            pickle.dump(cluster_list, output_file)
            output_file.close()
            file_count += 1

            gen_list  = []
            tc_list = []

    if len(tc_list) > 0:
        output_file = open(f'{output_dir}/output_{file_count}.pkl', 'wb')
        pickle.dump(gen_list, output_file)
        pickle.dump(tc_list, output_file)
        pickle.dump(cluster_list, output_file)
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
    parser.add_argument('output',
                        help='output destination directory',
                        type=str
                        )
    parser.add_argument('-p', '--processes',
                        help='number of concurrent processes to run',
                        default=8,
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
    filepath  = args.input
    n_process = args.processes
    filename  = filepath.split('/')[-1].split('.')[0]

    # multiprocess the data mixing and algorithm testing
    if not os.path.exists(args.output):
            os.makedirs(args.output)

    run_data    = dict(file_count = 0,
                       filepath   = filepath,
                       output_dir = args.output,
                       file_type  = args.file_type
                       ) 
    convert_tree(run_data)

    #if os.path.isfile(args.input):
    #    run_data    = dict(file_count = 0,
    #                       filepath   = filepath,
    #                       output_dir = args.output,
    #                       file_type  = args.file_type
    #                       ) 
    #    convert_tree(run_data)

    #elif os.path.isdir(args.input):
    #    pool = Pool(processes = n_process)
    #    for i, filename in enumerate(os.listdir(args.input)):
    #        if not filename.endswith('.root'): 
    #            continue

    #        run_data = dict(file_count = i,
    #                        filepath   = filepath + filename,
    #                        output_dir = args.output,
    #                        file_type  = args.file_type
    #                        ) 
    #        pool_result = pool.apply_async(convert_tree, args=(run_data))

    #    pool.close()
    #    pool.join()
