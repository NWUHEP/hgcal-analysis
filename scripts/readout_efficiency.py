#!/usr/bin/env python

import numpy as np
import pandas as pd
from root_pandas import read_root
import matplotlib.pyplot as plt
import time

def augment_ntuple(df, geom):
    z = df.tc_z
    eta = df.tc_eta
    phi = df.tc_phi
    theta = 2*np.arctan(np.exp(-eta))
    x = z*np.tan(theta)*np.cos(phi)
    y = z*np.tan(theta)*np.sin(phi)
    df['tc_x'] = x
    df['tc_y'] = y
    
    # motherboard mapping
    print('mapping trigger cells to motherboards')
    mapping_dict = pd.Series(geom.id.values, index=geom.tc_id).to_dict()
    df['tc_mboard'] = df.tc_id.map(mapping_dict)
    print('motherboard mapping complete')

    return df

def set_bunches(df_sig, df_bkg, n_sig, n_bkg):
    sig_evts = np.random.choice(df_sig.event.unique(), n_sig, replace=False)
    bkg_evts = np.random.choice(df_bkg.event.unique(), n_bkg, replace=False)
    sig_tc = df_sig[df_sig.event.isin(sig_evts)]
    bkg_tc = df_bkg[df_bkg.event.isin(bkg_evts)]
    df = bkg_tc.append(sig_tc)
    evtlist = df.event.unique()
    gen_energy = sig_tc.gen_energy.values[0]

    return (evtlist, gen_energy)

def test_best_choice(df, evtlist, params):
    #nbx = params[0]            # take to be same as len(evtlist) for now
    nwaferbx = params[1] 
    df_out = pd.DataFrame()
    df = df[df.event.isin(evtlist)]
    tagged_tc = df.query('tc_simenergy > 0')
    for layer in df.tc_layer.unique():
        this_df = df.query('tc_layer == {0}'.format(layer))
        this_tag = tagged_tc.query('tc_layer == {0}'.format(layer))
        tagged_mboards = this_tag.tc_mboard.unique()
        this_df = this_df[this_df.tc_mboard.isin(tagged_mboards)]
        map_tuples = list(zip(this_df.event, this_df.tc_mboard, this_df.tc_wafer))
        this_df['event_mboard_wafer'] = map_tuples
        groupby = this_df.groupby('event_mboard_wafer')
        wafer_sum_map = groupby.apply(lambda d: d.tc_energy.sum()).to_dict()
        wafer_tc_energy = this_df.event_mboard_wafer.map(wafer_sum_map)
        this_df.loc[:, 'wafer_tc_energy'] = wafer_tc_energy            
        sorted_wafer_sums = np.sort(this_df.wafer_tc_energy.unique())[::-1]
        if len(sorted_wafer_sums) >= nwaferbx:
            isReadout = this_df.wafer_tc_energy >= sorted_wafer_sums[nwaferbx-1]
        else:
            isReadout = np.ones(this_df.shape[0], dtype=bool)
        this_df.loc[:, 'tc_isReadout'] = isReadout
        this_df = this_df[['event', 'tc_layer', 'tc_wafer', 'tc_id', 'tc_isReadout', 
                              'tc_energy', 'tc_simenergy', 'wafer_tc_energy']]
        
        df_out = df_out.append(this_df)

    return df_out

def test_threshold(df, evtlist, params):
    thresh = params[0]
    df_out = pd.DataFrame()
    tagged_tc = df.query('tc_simenergy > 0')
    tagged_tc = tagged_tc[tagged_tc.event.isin(evtlist)]
    for layer in df.tc_layer.unique():
        this_tag = tagged_tc.query('tc_layer == {0}'.format(layer))
        tc_over_thresh = tagged_tc.tc_mipPt > thresh
        this_tag['isReadout'] = tc_over_thresh
        this_df = this_tag[['event', 'tc_layer', 'tc_wafer', 'tc_id', 'tc_isReadout', 'tc_energy', 'tc_simenergy']]
        df_out = df_out.append(this_df)

    return df_out
    
if __name__ == '__main__':
    
    # load trigger and geometry data
    print('reading dataframes')
    tc_features = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_wafer', 
                   'tc_cell', 'tc_energy',  'tc_simenergy', 'tc_mipPt', 'tc_eta', 'tc_phi', 'tc_z']

    tc_pu_features = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_wafer', 
                      'tc_cell', 'tc_energy', 'tc_eta', 'tc_phi', 'tc_z']

    gen_features = ['event', 'gen_id', 'gen_status', 'gen_energy', 'gen_pt', 'gen_eta', 'gen_phi']

    geom_features = ['id', 'zside', 'subdet','layer', 'module', 'x', 'y', 'z', 'tc_layer',
                     'tc_wafer', 'tc_zside', 'tc_subdet', 'tc_id', 'tc_x', 'tc_y', 'tc_z']

    df_tc_gamma = read_root('data/trig/ntuple_singleGamma140_newmap.root', 
                            'hgcalTriggerNtuplizer/HGCalTriggerNtuple', 
                             columns=tc_features, flatten=tc_features)

    df_tc_pu = read_root('data/trig/ntuple_doublenu140_newmap_nothresh.root', 
                         'hgcalTriggerNtuplizer/HGCalTriggerNtuple', 
                          columns=tc_pu_features, flatten=tc_pu_features)

    df_gen = read_root('data/trig/ntuple_singleGamma140_newmap.root', 
                       'hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                        columns=gen_features, flatten=gen_features)

    df_geom = read_root('data/geom/test_triggergeom_newmap.root', 
                        'hgcaltriggergeomtester/TreeModules', 
                         columns=geom_features, flatten=geom_features)
    print('reading dataframes complete')

    df_tc_gamma = pd.merge(df_tc_gamma, df_gen, on='event')
    df_tc_gamma.query('gen_status == 1 and gen_id == 22 and gen_eta > 0', inplace=True)
                         
    df_tc = df_tc_gamma.append(df_tc_pu)
    augment_ntuple(df_tc, df_geom)          # coordinates and wafer->motherboard mapping
    
    df_tc_gamma.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_tc_pu.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_tc.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_tc_gamma.fillna(0.0, inplace=True)
    df_tc_pu.fillna(0.0, inplace=True)
    df_tc.fillna(0.0, inplace=True)
    df_geom.query('zside == 1 and subdet == 3', inplace=True)

    # algorithm settings
    algo = 'best_choice'
    if algo == 'best_choice':
        nbx = 9     # not used (gets set to n_sig+n_bkg for now)
        nwaferbx = 4
        params = [nbx, nwaferbx]
    elif algo == 'threshold':
        threshold = 2. # measured in mipPt
        params = [threshold]
    
    # bunch arrangement settings
    n_sig = 1
    n_bkg = 8
    
    nruns = 1

    df_out = pd.DataFrame()
    for iRun in range(nruns):
        (evtlist, gen_energy) = set_bunches(df_tc_gamma, df_tc_pu, n_sig, n_bkg)
        df_algo = pd.DataFrame()
        if algo == 'best_choice':
            df_algo = test_best_choice(df_tc, evtlist, params)
        elif algo == 'threshold':
            df_algo = test_threshold(df_tc, evtlist, params)
        df_algo['run'] = iRun
        df_algo['gen_energy'] = gen_energy
        df_out = df_out.append(df_algo)
        print('{0} % complete'.format(100.*iRun/nruns))    

    df_out.to_csv('data/eff/readout_eff_test.csv', index=False)
