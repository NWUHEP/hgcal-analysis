#!/usr/bin/env python

import numpy as np
import pandas as pd
import time
from root_pandas import read_root
import ROOT as r
import sys

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

    return df

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
        gby = this_df.query('tc_simenergy > 0.').groupby('event')
        eta_map = gby.apply(lambda d: d.tc_eta.mean()).to_dict()
        phi_map = gby.apply(lambda d: d.tc_phi.mean()).to_dict()
        this_df.loc[:, 'simeta_mean'] = this_df.event.map(eta_map).fillna(1000)
        this_df.loc[:, 'simphi_mean'] = this_df.event.map(phi_map).fillna(1000)
        eta_window = 0.05
        phi_window = 0.2
        in_window = np.logical_and(np.less(abs(this_df.tc_eta - this_df.simeta_mean), eta_window), np.less(abs(this_df.tc_phi - this_df.simphi_mean), phi_window))
        this_df.loc[:, 'in_window'] = in_window
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
        this_df.query('tc_simenergy > 0 or in_window == True', inplace=True)
        
        df_out = df_out.append(this_df)

    return df_out

def test_threshold(df, params):
    thresh = params[0]
    df_out = pd.DataFrame()
    for layer in df.tc_layer.unique():
        this_tag = df.query('tc_layer == {0}'.format(layer))
        tc_over_thresh = this_tag.tc_mipPt > thresh
        this_tag['tc_isReadout'] = tc_over_thresh
        gby = this_tag.query('tc_simenergy > 0.').groupby('event')
        eta_map = gby.apply(lambda d: d.tc_eta.mean()).to_dict()
        phi_map = gby.apply(lambda d: d.tc_phi.mean()).to_dict()
        this_tag.loc[:, 'simeta_mean'] = this_tag.event.map(eta_map).fillna(1000)
        this_tag.loc[:, 'simphi_mean'] = this_tag.event.map(phi_map).fillna(1000)
        eta_window = 0.05
        phi_window = 0.2
        in_window = np.logical_and(np.less(abs(this_tag.tc_eta - this_tag.simeta_mean), eta_window), np.less(abs(this_tag.tc_phi - this_tag.simphi_mean), phi_window))
        this_tag.loc[:, 'in_window'] = in_window
        this_tag.query('tc_simenergy > 0 or in_window == True', inplace=True)
        df_out = df_out.append(this_tag)

    return df_out
    
if __name__ == '__main__':
    
    job_id = sys.argv[1]
    output_name = 'singleGamma_pt25_pu140_' + job_id + '.csv'

    #path = '/eos/uscms/store/user/jbueghly/hgcal/'
    path = ''

    # import event lists
    evt_df = pd.read_pickle(path+'evts.pkl')
    sig_evts = evt_df.sig_evts.values
    bkg_evts = evt_df.bkg_evts.values

    # load trigger data
    print('reading dataframes')
    tc_features = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_wafer', 
                   'tc_energy', 'tc_mipPt', 'tc_eta', 'tc_phi', 'tc_z']
 
    gen_features = ['event', 'gen_id', 'gen_status', 'gen_energy', 'gen_eta'] 
    
    # algorithm settings
    algo = 'best_choice'
    if algo == 'best_choice':
        nbx = 9     # not used (gets set to n_sig+n_bkg for now)
        nwaferbx = 4
        params = [nbx, nwaferbx]
    elif algo == 'threshold':
        threshold = 2. # measured in mipPt
        params = [threshold]

    n_sig = 1
    n_bkg = 9
    nruns = 100

    df_out = pd.DataFrame()
    for i in range(nruns):

        sig_list = np.random.choice(sig_evts, n_sig, replace=False)
        bkg_list = np.random.choice(bkg_evts, n_bkg, replace=False)
        sig_str = '|'.join(['event == {0}'.format(evt) for evt in sig_list])
        bkg_str = '|'.join(['event == {0}'.format(evt) for evt in bkg_list])

        df_tc_sig = read_root(path+'ntuple_singleGamma_1000.root',
                             'hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                              columns=tc_features+['tc_simenergy'],
                              flatten=tc_features+['tc_simenergy'],
                              where=sig_str)
        
        df_gen    = read_root(path+'ntuple_singleGamma_1000.root',
                             'hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                              columns=gen_features,
                              flatten=gen_features,
                              where=sig_str)
        
        df_tc_bkg = read_root(path+'ntuple_doublenu_1000.root',
                               'hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                                columns=tc_features,
                                flatten=tc_features,
                                where=bkg_str)

        df_tc_sig = pd.merge(df_tc_sig, df_gen, on='event')
        df_tc_sig.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
        df_tc_sig.query('gen_status == 1 and gen_id == 22 and gen_eta > 0', inplace=True)
        
        df_tc_bkg.query('tc_zside == 1 and tc_subdet == 3', inplace=True)

        trig = df_tc_bkg.append(df_tc_sig)
        trig.loc[:, 'run'] = int(job_id)*nruns + i
        
        if algo == 'best_choice':
            trig = test_best_choice(trig, params)
        elif algo == 'threshold':
            trig = test_threshold(trig, params)

        df_out = df_out.append(trig)
        print('{0} % complete'.format(100.*(i+1)/nruns))    
    
    df_geom = pd.read_pickle(path+'geom.pkl')
    df_geom.query('zside == 1 and subdet == 3', inplace=True)
    augment_ntuple(df_out, df_geom)          

    df_out.fillna(0, inplace=True)

    df_int = df_out.select_dtypes(include=['int'])
    df_out = df_out.apply(pd.to_numeric, downcast='unsigned')
    df_float = df_out.select_dtypes(include=['float'])
    df_out = df_out.apply(pd.to_numeric, downcast='float')

    print('writing dataframe')
    df_out.to_csv(output_name)
    print('dataframe written successfully')
