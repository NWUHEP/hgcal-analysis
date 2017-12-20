#!/usr/bin/env python

import numpy as np
import pandas as pd
from root_pandas import read_root
import matplotlib.pyplot as plt
import time

tc_features = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_wafer', 
               'tc_cell', 'tc_energy',  'tc_simenergy', 'tc_eta', 'tc_phi', 'tc_z']

tc_pu_features = ['event', 'tc_id', 'tc_subdet', 'tc_zside', 'tc_layer', 'tc_wafer', 
                  'tc_cell', 'tc_energy', 'tc_eta', 'tc_phi', 'tc_z']

gen_features = ['event', 'gen_id', 'gen_status', 'gen_energy', 'gen_pt', 'gen_eta', 'gen_phi']

geom_features = ['id', 'zside', 'subdet','layer', 'module', 'x', 'y', 'z', 'tc_layer',
                 'tc_wafer', 'tc_zside', 'tc_subdet', 'tc_id', 'tc_x', 'tc_y', 'tc_z']

df_tc_gamma = read_root('data/ntuple_singleGamma140_newmap.root', 
                        'hgcalTriggerNtuplizer/HGCalTriggerNtuple', 
                         columns=tc_features, flatten=tc_features)

df_tc_pu = read_root('data/ntuple_doublenu140_newmap.root', 
                     'hgcalTriggerNtuplizer/HGCalTriggerNtuple', 
                      columns=tc_pu_features, flatten=tc_pu_features)


df_gen = read_root('data/ntuple_singleGamma140_newmap.root', 
                   'hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                    columns=gen_features, flatten=gen_features)

df_geom = read_root('data/test_triggergeom_newmap.root', 'hgcaltriggergeomtester/TreeModules', 
                     columns=geom_features, flatten=geom_features)

df_tc_gamma = pd.merge(df_tc_gamma, df_gen, on='event')
df_tc_gamma = df_tc_gamma.query('gen_status == 1 and gen_id == 22 and gen_eta > 0')
                     
df_tc = df_tc_gamma.append(df_tc_pu)
z = df_tc.tc_z
eta = df_tc.tc_eta
phi = df_tc.tc_phi
theta = 2*np.arctan(np.exp(-eta))
x = z*np.tan(theta)*np.cos(phi)
y = z*np.tan(theta)*np.sin(phi)
df_tc['tc_x'] = x
df_tc['tc_y'] = y

print('mapping trigger cells to motherboards')
start = time.time()
mapping_dict = pd.Series(df_geom.id.values, index=df_geom.tc_id).to_dict()
df_tc['tc_mboard'] = df_tc.tc_id.map(mapping_dict)
end = time.time()
print('motherboard mapping complete')
print('execution time = {0}'.format(end-start))

df_tc_gamma = df_tc_gamma.query('tc_zside == 1 and tc_subdet == 3')
df_tc_pu = df_tc_pu.query('tc_zside == 1 and tc_subdet == 3')
df_tc = df_tc.query('tc_zside == 1 and tc_subdet == 3')

df_geom = df_geom.query('zside == 1 and subdet == 3')

# tagging photon sim energy
tagged_tc = df_tc.query('tc_simenergy > 0')

# probing with JB's readout 
nbx = 9 #readout param
nwaferbx = 4 #readout param
nruns = 1

df_out = pd.DataFrame()
for iRun in range(nruns):
    mini_tc = df_tc_gamma.sample()
    mini_pu = df_tc_pu.sample(nbx-1)
    mini_df = mini_pu.append(mini_tc).sample(frac=1)
    evtlist = mini_df.event.unique()
    gen_energy = mini_tc.gen_energy.values[0]
    for layer in df_tc.tc_layer.unique():
        print('Run {0}, layer {1}'.format(iRun, layer))
        this_df = df_tc[df_tc.event.isin(evtlist)].query('tc_layer == {0}'.format(layer))
        this_tag = tagged_tc[tagged_tc.event.isin(evtlist)].query('tc_layer == {0}'.format(layer))
        tagged_mboards = this_tag.tc_mboard.unique()
        mboard_gby = this_df.groupby('tc_mboard')
        for mboard in tagged_mboards:
            this_group = mboard_gby.get_group(mboard)
            groupby = this_group.groupby(['event', 'tc_wafer'])
            df_sum = groupby.sum().sort_values(by=['tc_energy'], ascending=False).reset_index()
            df_sum['isReadout'] = False
            df_sum.isReadout[:nwaferbx] = True
            Ndf = len(df_sum)
            this_df = pd.DataFrame({'run': [iRun]*Ndf, 'layer': [layer]*Ndf, 'wafer': df_sum.tc_wafer, 'isReadout': df_sum.isReadout, 
                                    'tc_energy': df_sum.tc_energy, 'tc_simenergy': df_sum.tc_simenergy, 'gen_energy': [gen_energy]*Ndf})
            df_out = df_out.append(this_df)

df_out.to_csv('data/eff/readout_eff.csv', index=False)
