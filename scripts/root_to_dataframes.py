#!/usr/bin/env python

import numpy as np
import pandas as pd
from root_pandas import read_root

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
    augment_ntuple(df_tc_gamma, df_geom)          
    augment_ntuple(df_tc_pu, df_geom)

    # basic cuts/processing
    df_tc_gamma.query('gen_status == 1 and gen_id == 22 and gen_eta > 0', inplace=True)
    df_tc_gamma.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_tc_pu.query('tc_zside == 1 and tc_subdet == 3', inplace=True)
    df_tc_gamma.fillna(0.0, inplace=True)
    df_tc_pu.fillna(0.0, inplace=True)
    df_geom.query('zside == 1 and subdet == 3', inplace=True)

    print('pickling dataframes')
    df_tc_gamma.to_pickle('data/flattuples/singleGamma_pt25_pu140.pkl')
    df_tc_pu.to_pickle('data/flattuples/doublenu_pu140.pkl')
    df_geom.to_pickle('data/flattuples/geom.pkl')
    print('dataframes pickled successfully')
