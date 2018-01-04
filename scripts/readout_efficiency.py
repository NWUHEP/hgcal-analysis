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
    sig_tc = df_sig.sample(n_sig)
    bkg_tc = df_bkg.sample(n_bkg)
    df = bkg_tc.append(sig_tc).sample(frac=1)
    evtlist = df.event.unique()
    gen_energy = sig_tc.gen_energy.values[0]

    return (evtlist, gen_energy)

def test_best_choice(df, evtlist, params):
    #nbx = params[0]            # take to be same as len(evtlist) for now
    nwaferbx = params[1] 
    df_out = pd.DataFrame()
    tagged_tc = df.query('tc_simenergy > 0')
    for layer in df.tc_layer.unique():
        this_df = df[df.event.isin(evtlist)].query('tc_layer == {0}'.format(layer))
        this_tag = tagged_tc[tagged_tc.event.isin(evtlist)].query('tc_layer == {0}'.format(layer))
        tagged_mboards = this_tag.tc_mboard.unique()
        mboard_gby = this_df.groupby('tc_mboard')
        for mboard in tagged_mboards:
            this_group = mboard_gby.get_group(mboard)
            groupby = this_group.groupby(['event', 'tc_wafer'])
            df_sum = groupby.sum().sort_values(by=['tc_energy'], ascending=False).reset_index()
            isReadout = np.zeros(df_sum.shape[0], dtype=bool)
            isReadout[:nwaferbx] = True
            df_sum['isReadout'] = isReadout
            this_df = pd.DataFrame({'layer': [layer]*df_sum.shape[0], 'wafer': df_sum.tc_wafer, 'isReadout': df_sum.isReadout, 
                                    'tc_energy': df_sum.tc_energy, 'tc_simenergy': df_sum.tc_simenergy})
            df_out = df_out.append(this_df)

    return df_out


if __name__ == '__main__':
    
    # load trigger and geometry data
    print('reading dataframes')
    tc_features = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_wafer', 
                   'tc_cell', 'tc_energy',  'tc_simenergy', 'tc_eta', 'tc_phi', 'tc_z']

    tc_pu_features = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_wafer', 
                      'tc_cell', 'tc_energy', 'tc_eta', 'tc_phi', 'tc_z']

    gen_features = ['event', 'gen_id', 'gen_status', 'gen_energy', 'gen_pt', 'gen_eta', 'gen_phi']

    geom_features = ['id', 'zside', 'subdet','layer', 'module', 'x', 'y', 'z', 'tc_layer',
                     'tc_wafer', 'tc_zside', 'tc_subdet', 'tc_id', 'tc_x', 'tc_y', 'tc_z']

    df_tc_gamma = read_root('data/trig/ntuple_singleGamma140_newmap.root', 
                            'hgcalTriggerNtuplizer/HGCalTriggerNtuple', 
                             columns=tc_features, flatten=tc_features)

    df_tc_pu = read_root('data/trig/ntuple_doublenu140_newmap.root', 
                         'hgcalTriggerNtuplizer/HGCalTriggerNtuple', 
                          columns=tc_pu_features, flatten=tc_pu_features)

    df_gen = read_root('data/trig/ntuple_singleGamma140_newmap.root', 
                       'hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                        columns=gen_features, flatten=gen_features)

    df_geom = read_root('data/geom/test_triggergeom_newmap.root', 'hgcaltriggergeomtester/TreeModules', 
                         columns=geom_features, flatten=geom_features)
    print('reading dataframes complete')

    df_tc_gamma = pd.merge(df_tc_gamma, df_gen, on='event')
    df_tc_gamma.query('gen_status == 1 and gen_id == 22 and gen_eta > 0', inplace=True)
                         
    df_tc = df_tc_gamma.append(df_tc_pu)
    augment_ntuple(df_tc, df_geom)          # coordinates and wafer->motherboard mapping
    
    df_tc_gamma.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_tc_pu.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_tc.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_geom.query('zside == 1 and subdet == 3', inplace=True)

    # algorithm settings
    algo = 'best_choice'
    if algo == 'best_choice':
        nbx = 9     # not used (gets set to n_sig+n_bkg for now)
        nwaferbx = 4
        params = [nbx, nwaferbx]
    elif algo == 'threshold':
        threshold = 0.
        params = [threshold]
    
    # bunch arrangement settings
    n_sig = 1
    n_bkg = 8
    
    nruns = 5
    df_out = pd.DataFrame()
    for iRun in range(nruns):
        (evtlist, gen_energy) = set_bunches(df_tc_gamma, df_tc_pu, n_sig, n_bkg)
        df_algo = test_best_choice(df_tc, evtlist, params)
        df_algo['run'] = iRun
        df_algo['gen_energy'] = gen_energy
        df_out = df_out.append(df_algo)
        print('{0} % complete'.format(100.*iRun/nruns))

    df_out.to_csv('data/eff/readout_eff.csv', index=False)
