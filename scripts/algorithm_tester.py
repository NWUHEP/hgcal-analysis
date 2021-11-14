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

if __name__ == '__main__':

    # parse arguments #
    parser = argparse.ArgumentParser(description='Convert hgcal root ntuples to dataframes')
    parser.add_argument('input_dir',
                        help='root file containing signal process',
                        type=str
                        )
    parser.add_argument('-p', '--processes',
                        help='number of concurrent processes to run',
                        default=8,
                        type=int
                        )
    args = parser.parse_args()
    ###############################

    # unpack arguments
    gen_list = []
    df_list = []
    inputdir = 'data/mc_mixtures/single_electron_pt35_skim_20180328_154138'
    for filename in os.listdir(inputdir):
        data_file = open(f'{inputdir}/{filename}', 'rb')
        gen_list.extend(pickle.load(data_file))
        df_list.extend(pickle.load(data_file))
        data_file.close()

    algo_list = ['baseline', 
                 'threshold_1bx_esort',  'threshold_1bx_nosort', 
                 'threshold_8bx_esort',  'threshold_8bx_nosort'
                ]
    mippt_scan = np.arange(0, 10, 1)
    ratios = {n:{mippt:[] for mippt in mippt_scan} for n in algo_list}
    cell_labels = ['zside', 'layer', 'sector', 'panel']
    data_iter = list(zip(gen_list, df_list))
    for gpart, data in tqdm_notebook(data_iter, total=len(data_iter)):
        gen1 = gpart.iloc[0]
        gen2 = gpart.iloc[1]
        denom = gen1.pt
        
        # carry out algorithms

    # multiprocess the data mixing and algorithm testing
    #output_dir = f'data/mc_mixtures/{input_name}_{get_current_time()}'
    #if not os.path.exists(output_dir):
    #        os.makedirs(output_dir)

    #run_data    = [dict(file_count  = i,
    #                    n_epochs    = n_epochs,
    #                    signal_filename = signal_filename,
    #                    pileup_filename = pileup_filename,
    #                    output_dir  = output_dir
    #                    ) for i in range(n_process)]
    #pool        = Pool(processes = n_process)
    #pfunc       = partial(mix_and_analyze)
    #pool_result = pool.map(pfunc, run_data)

    #pool.close()
    #pool.join()
    #pool.close()
