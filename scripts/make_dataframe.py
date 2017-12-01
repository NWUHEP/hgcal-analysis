#!/usr/bin/env python

from TriggerUtility import *
from root_pandas import read_root
import time

def root_to_dataframe(path, tree, features, index):
    '''
    Converts a semi-flat root ntuple to a pandas dataframe using root_pandas.

    Parameters:
    ===========
    path : location of input ROOT file
    features : these are the branches to be put into the dataframe.  These
               should be vectors of basic types which will be flattened.
    index : the column to use as the first index (the second index will be
            __array_index that comes out of the root_pandas flattening).
    '''

    df = read_root(path, tree, columns=[index]+features, flatten=features)
    df.index = df[index]
    df = df[features]
    return df


features = ['tc_id', 'tc_zside', 'tc_subdet', 'tc_layer', 'tc_wafer', 'tc_wafertype',
            'tc_eta', 'tc_phi', 'tc_z', 'tc_energy']

geom = read_root("data/test_triggergeom.root","hgcaltriggergeomtester/TreeTriggerCells")
trig = root_to_dataframe("data/ntuple.root","hgcalTriggerNtuplizer/HGCalTriggerNtuple", features, index='event')

flatlist = []
labels = ['zside', 'layer', 'subdet', 'evt', 'mboard_id', 'mod_id', 'tc_id', 'tc_energy']

for zside in [1]: # only positive z for now
    print(zside)
    this_geom = geom.query('zside == {0}'.format(zside))
    for subdet in [3]: # only ECal for now
        print(subdet)
        this_subgeom = this_geom.query('subdet == {0}'.format(subdet))
        for layer in this_subgeom.layer.unique()[:1]: # only one layer for now
            print(layer)
            start = time.time()
            geom_map = loadtriggermapping(layer,zside,"data/test_triggergeom.root",subdet,
                                          "data/geom_with_motherboard.pkl") 
            this_trig = trig.query('tc_zside == {0} and tc_layer == {1} and tc_subdet == {2}'.format(zside, layer, subdet))
            for mboard in geom_map.mboard_id.unique():
                this_df = geom_map.query('mboard_id == {0}'.format(mboard))
                for mod in this_df.mod_id.unique():
                    this_subdf = this_df.query('mod_id == {0}'.format(mod));
                    for tc in this_subdf.tc_id.unique():
                        this_subtrig = this_trig.loc[this_trig['tc_id'] == tc]
                        for evt in this_trig.index.unique():
                            if not (evt in this_subtrig.index.unique()):
                                energy = 0.
                            else:
                                this_evt = this_subtrig.loc[evt]
                                energy = this_evt.tc_energy
                            row = [zside, layer, subdet, evt, mboard, mod, tc, energy]
                            flatlist.append(row)
            finish = time.time()
            print(finish - start)

df_out = pd.DataFrame.from_records(flatlist, columns=labels)
df_out = df_out.reset_index()
df_out.to_pickle('data/tc_energies.pkl')
