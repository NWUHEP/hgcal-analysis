'''
    Matches trigger cells to reconstructed clusters and gen particles, and
    converts input root files to dataframes.
'''

import os
import sys
import argparse
import subprocess
import warnings
from itertools import chain
from pathlib import Path
from tempfile import TemporaryFile

from tqdm import tqdm
import pickle
import yaml
import awkward
import uproot
import numpy as np
import h5py
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def set_indices(df):
    # modifies multiindex from uproot so that the leading index corresponds the
    # event number
    index = df.index
    event_numbers = df['event']
    new_index = [(e, ix[1]) for e, ix in zip(event_numbers, index)]
    df.set_index(pd.MultiIndex.from_tuples(new_index, names=['event', 'id']), inplace=True)
    df.drop('event', axis=1, inplace=True)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config',
                        default='config/tc_matching_cfg.yaml',
                        help='File specifying configuration of matching process.'
                        )
    parser.add_argument('--job_id', 
                        default=0, 
                        type=int, 
                        help='Index of the output job.'
                        )
    parser.add_argument('--input_file', 
                        default=None, 
                        type=str, 
                        help='File with list of input files.  If this option is used, it will override the value provided in the configuration file.'
                        )
    parser.add_argument('--output_dir', 
                        default=None, 
                        type=str, 
                        help='Directory to write output file to.  If this option is used, it will override the value provided in the configuration file.'
                        )
    parser.add_argument('--max_events', 
                        default=None, 
                        type=int, 
                        help='Maximum number of events to process.  Useful for testing...'
                        )
    parser.add_argument('--events_per_file', 
                        default=100, 
                        type=int, 
                        help='Set the maximum number of events to be saved to each output file.  The value should be selected to prevent using more memory than available.'
                        )
    parser.add_argument('--is_batch', 
                        action='store_true',
                        help='Use this if running with a (lpc condor?) batch system to enable xrd to open remote files.'
                        )
    args = parser.parse_args()
   
    # Load configuration file
    with open(args.config, 'r') as config_file: 
        config = yaml.safe_load(config_file)
    
    # Unpack options from configuration file
    dr_threshold    = config['dr_threshold']
    gen_tree_name   = config['gen_tree']
    match_only      = config['match_only']
    reached_ee      = config['reached_ee']
    backend         = config['backend']
    frontend_algos  = config['frontends']
    ntuple_template = config['ntuple_template']
    output_dir      = config['output_destination']

    branches_gen    = config['branches_gen']
    branches_cl3d   = config['branches_cl3d']
    branches_tc     = config['branches_tc']

    # baseline cuts (move these to the configuration file?)
    gen_cuts     = f'(genpart_reachedEE == {reached_ee}) and (genpart_gen != -1)'
    tc_cuts      = 'tc_energy > 0.01'
    cluster_cuts = 'cl3d_pt > 5.'

    if args.input_file:
        with open(args.input_file) as f:
            file_list = f.read().splitlines()
    else:
        if type(config['input_files']) == str:
            with open(config['input_files'], 'r') as f:
                file_list = f.read().splitlines()
        else:
            file_list = config['input_files']

    if args.output_dir:
        output_dir = args.output_dir

    if args.is_batch: # having some issues with xrd on condor
        # override some user options when running over batch
        uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource
        output_dir = 'data'

    print('Getting gen particles and trigger cells...')

    # read root files
    df_gen_list = []
    layer_labels = [f'cl3d_layer_{n}_pt' for n in range(36)]
    for filename in tqdm(file_list, desc='Processing files and retrieving data...'):
        tqdm.write(filename)

        # get gen particles
        uproot_file = uproot.open(filename)
        gen_tree = uproot_file[gen_tree_name]
        df_gen = pd.concat([df for df in gen_tree.iterate(branches_gen, library='pd', step_size=args.events_per_file, entry_stop=args.max_events)])
        df_gen.query(gen_cuts, inplace=True)
        df_gen_list.append(df_gen)
        df_gen = pd.concat(df_gen_list)
        set_indices(df_gen)

        # get trigger cells (do this for the most inclusive algorithm, same as gen tree)
        tc_tree = uproot_file[gen_tree_name]
        tree_iter = tc_tree.iterate(branches_tc, 
                                    #cut=tc_cuts,
                                    library='pd', 
                                    step_size=args.events_per_file, 
                                    entry_stop=args.max_events
                                    )
        for i, df_tc in tqdm(enumerate(tree_iter)):

            # apply some cuts
            if tc_cuts != '':
                df_tc.query(tc_cuts, inplace=True)

            # set indices to (event, tc_id)
            set_indices(df_tc)
    
            ### save files to savedir in HDF (temporarily use pickle files because of problems with hdf5 on condor)
            output_name = f'{output_dir}/output_{args.job_id}_{i}.pkl'
            outfile = open(output_name, 'wb')
            output_dict = dict(gen=df_gen, tc=df_tc)
            pickle.dump(output_dict, outfile)
            outfile.close()
            tqdm.write(f'Writing output to {output_name}')

            # training data: write the data from each wafer above some energy
            # into a 14x8x8 grid (layer x cellu x cellv).  Save data to
            # two-level dictionaries where the first level has keys (event
            # number, zside) and the next level has (waferu, waferv).  In the
            # next iteration, introduce rotating the waferu, waferv so that the
            # maximum wafer lies in the first sextent (between 0 and pi/3).

            # First identify wafers with some minimum energy treshold to save
            group_labels      = ['event', 'tc_zside', 'tc_layer', 'tc_waferu', 'tc_waferv']
            df_tcee          = df_tc.query('tc_subdet == 1')
            g_wafers          = df_tcee.groupby(group_labels)
            wafer_sums        = g_wafers.apply(lambda x: x['tc_energy'].sum()) 
            masked_wafer_sums = wafer_sums.loc[wafer_sums > 1.] # only keep wafers with enough energy to be interesting

            # in the next iteration, I will save wafers by layers, for now just
            # keep all wafers and don't differentiate based on layer or wafer
            # id 
            
            group_labels.remove('event')
            df_tcee = df_tcee.set_index(group_labels, append=True).droplevel(1)
            data_dict = dict()
            for ix_wafer in masked_wafer_sums.index:
                event, zside = ix_wafer[0], ix_wafer[1]
                if (event, zside) not in data_dict.keys():
                    data_dict[(event, zside)] = dict()

                wafer_dict = data_dict[(event, zside)]
                waferu, waferv = ix_wafer[3], ix_wafer[4]
                if (waferu, waferv) not in wafer_dict.keys():
                    wafer_dict[(waferu, waferv)] = np.zeros((14, 8, 8))

                wafer_grid = wafer_dict[(waferu, waferv)]
                df_wafer = df_tcee.loc[ix_wafer]
                df_wafer.set_index(['tc_cellu', 'tc_cellv'], inplace=True)
                for (cellu, cellv), e in df_wafer['tc_energy'].items():
                    layer = int((ix_wafer[2] - 1)/2)
                    wafer_grid[layer, cellu, cellv] = e

            output_name = f'{output_dir}/grids_{args.job_id}_{i}.pkl'
            outfile = open(output_name, 'wb')
            pickle.dump(data_dict, outfile)
            outfile.close()
            tqdm.write(f'Writing output to {output_name}')


