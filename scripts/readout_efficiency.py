#!/usr/bin/env python

import numpy as np
import pandas as pd

def set_bunches(df_sig, df_bkg, n_sig, n_bkg):
    sig_evts = np.random.choice(df_sig.event.unique(), n_sig, replace=False)
    bkg_evts = np.random.choice(df_bkg.event.unique(), n_bkg, replace=False)
    sig_tc = df_sig[df_sig.event.isin(sig_evts)]
    bkg_tc = df_bkg[df_bkg.event.isin(bkg_evts)]
    df = bkg_tc.append(sig_tc)
    gen_energy = sig_tc.gen_energy.values[0]

    return (df, gen_energy)

def test_best_choice(df, params):
    #nbx = params[0]            # take to be same as len(evtlist) for now
    nwaferbx = params[1] 
    df_out = pd.DataFrame()
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
        this_df = this_df[['event', 'tc_layer', 'tc_wafer', 'tc_id', 'tc_eta', 'tc_phi',
                           'tc_isReadout', 'tc_energy', 'tc_simenergy', 'wafer_tc_energy']]
        
        df_out = df_out.append(this_df)

    return df_out

def test_threshold(df, params):
    thresh = params[0]
    df_out = pd.DataFrame()
    tagged_tc = df.query('tc_simenergy > 0')
    for layer in df.tc_layer.unique():
        this_tag = tagged_tc.query('tc_layer == {0}'.format(layer))
        tc_over_thresh = tagged_tc.tc_mipPt > thresh
        this_tag['tc_isReadout'] = tc_over_thresh
        this_df = this_tag[['event', 'tc_layer', 'tc_wafer', 'tc_id', 'tc_eta', 'tc_phi', 
                            'tc_isReadout', 'tc_energy', 'tc_simenergy']]
        df_out = df_out.append(this_df)

    return df_out
    
if __name__ == '__main__':
    
    # load trigger and geometry data
    print('reading dataframes')
    df_tc_gamma = pd.read_pickle('data/flattuples/singleGamma_pt25_pu140.pkl')
    df_tc_pu = pd.read_pickle('data/flattuples/doublenu_pu140.pkl')
    df_geom = pd.read_pickle('data/flattuples/geom.pkl')
    print('reading dataframes complete')

    # algorithm settings
    algo = 'threshold'
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
    
    nruns = 1000

    df_out = pd.DataFrame()
    for iRun in range(nruns):
        (df_run, gen_energy) = set_bunches(df_tc_gamma, df_tc_pu, n_sig, n_bkg)
        df_algo = pd.DataFrame()
        if algo == 'best_choice':
            df_algo = test_best_choice(df_run, params)
        elif algo == 'threshold':
            df_algo = test_threshold(df_run, params)
        df_algo['run'] = iRun
        df_algo['gen_energy'] = gen_energy
        df_out = df_out.append(df_algo)
        print('{0} % complete'.format(100.*iRun/nruns))    

    df_out.to_csv('data/eff/readout_eff_test.csv', index=False)
