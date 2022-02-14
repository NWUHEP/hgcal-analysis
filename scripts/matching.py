
import os
import sys
import argparse
import subprocess
import warnings
from itertools import chain
from pathlib import Path

import pickle
import yaml
import awkward
import uproot
import numpy as np
import h5py
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def matching(event):
    return event.cl3d_pt==event.cl3d_pt.max()

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
                        default='config/matching_cfg.yaml',
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
    branches_gen    = config['branches_gen']
    branches_cl3d   = config['branches_cl3d']
    output_dir      = config['output_destination']

    layer_labels = [f'cl3d_layer_{n}_pt' for n in range(36)]
    gen_cuts = f'(genpart_reachedEE == {reached_ee}) and (genpart_gen != -1)'
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
        uproot.open.defaults["xrootd_handler"] = uproot.MultithreadedXRootDSource
        output_dir = 'data'

    # read root files
    df_gen_list = []
    dict_algos = {fe:[] for fe in frontend_algos}
    for filename in file_list:
        print(filename)
        uproot_file = uproot.open(filename)
        gen_tree = uproot_file[gen_tree_name]
        df_gen = pd.concat([df for df in gen_tree.iterate(branches_gen, library='pd', step_size=500)])
        df_gen.query(gen_cuts, inplace=True)
        df_gen_list.append(df_gen)

        for fe in frontend_algos:
            tree_name = ntuple_template.format(fe=fe, be=backend)
            algo_tree = uproot_file[tree_name]
            df_algo = pd.concat([df for df in algo_tree.iterate(branches_cl3d, library='pd', step_size=500)])

            # Trick to read layers pTs, which is a vector of vector
            algo_tree = uproot_file[tree_name]
            layer_pt = list(chain.from_iterable(algo_tree.arrays(['cl3d_layer_pt'])['cl3d_layer_pt'].tolist()))
            df_layer_pt = pd.DataFrame(layer_pt, columns=layer_labels, index=df_algo.index)
            df_algo = pd.concat([df_algo, df_layer_pt], axis=1)
          
            dict_algos[fe].append(df_algo)

    # concatenate dataframes for each algorithm after running over all files
    # clean particles that are not generator-level (genpart_gen) or didn't
    # reach endcap (genpart_reachedEE)
    df_gen = pd.concat(df_gen_list)
    set_indices(df_gen)
    df_gen_pos, df_gen_neg = [df for _, df in df_gen.groupby(df_gen['genpart_exeta'] < 0)]

    output_name = f'{output_dir}/output_{args.job_id}.pkl'
    outfile = open(output_name, 'wb')
    #store = pd.HDFStore(output_name, mode='w')
    output_dict = dict(gen=df_gen)
    for algo_name, dfs in dict_algos.items():
        df_algo = pd.concat(dfs)
        set_indices(df_algo)

        # calculate delta_r between clusters and gen particles and append
        # delta_r and associated gen properties for closest match (this could
        # use some cleanup)
        matched_features = []
        for (event, cl_id), cluster in df_algo.iterrows():
            cluster_eta, cluster_phi = cluster['cl3d_eta'], cluster['cl3d_phi']

            # first check if there are any gen particles passing quality cuts
            # with same eta sign as the cluster.  If there is no candidate for
            # matching, fill the entry with dummy values.
            if cluster_eta > 0:
                if df_gen_pos.index.isin([event], level=0).any():
                    df_gen_event = df_gen_pos.loc[event]
                else:
                    matched_features.append([-1, -1, -1])
                    continue
            else:
                if df_gen_neg.index.isin([event], level=0).any():
                    df_gen_event = df_gen_neg.loc[event]
                else:
                    matched_features.append([-1, -1, -1])
                    continue

            gen_eta = df_gen_event['genpart_exeta'].values
            gen_phi = df_gen_event['genpart_exphi'].values

            # calculate dr between cluster and all gen particles
            deta = cluster_eta - gen_eta
            dphi = cluster_phi - gen_phi
            dphi -= (dphi > np.pi)*2*np.pi
            dr = np.sqrt(deta*deta + dphi*dphi)

            # find index of gen particle with smallest value of dr and assign cluster its properties
            ix_min = dr.argmin()
            gen_candidate = df_gen_event.iloc[ix_min]
            matched_features.append([dr[ix_min], gen_candidate['genpart_pt'], gen_candidate['genpart_pid']])

        # save information about the per object gen matched quantities
        matched_features = np.array(matched_features)
        df_algo['matched_dr'] = matched_features[:, 0]
        df_algo['matched_pt'] = matched_features[:, 1]
        df_algo['matched_pid'] = matched_features[:, 2]
        
        # keep matched clusters only and select deltar under threshold
        # (threshold is set in configuration file)
        if match_only:
            df_algo.query(f'delta_r <= {dr_threshold} and delta_r != -1', inplace=True)
    
        output_dict[algo_name] = df_algo
        #store[algo_name] = df_algo

    ###save files to savedir in HDF (temporarily use pickle files because of problems with hdf5 on condor)
    pickle.dump(output_dict, outfile)
    outfile.close()
    print(f'Writing output to {output_name}')

    #store['gen'] = df_gen
    #store.close()
