
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from root_pandas import read_root
from tqdm import tqdm

import scripts.helpers as hlp

if __name__ == '__main__':

    panel_offset  = 0
    panel_mask    = 0x1F
    sector_offset = 5
    sector_mask   = 0x7
    features = ['id', 'zside', 'subdet', 'layer', 'module',
                #'x', 'y', 'z',
                'tc_layer', 'tc_zside', 'tc_subdet', 'tc_id',
                'tc_x', 'tc_y'
                ]

    df_new = read_root('data/test_triggergeom_newmap.root', 'hgcaltriggergeomtester/TreeModules',
                       columns=features, flatten=features)
    df_new['panel']  = [(val >> 8) & panel_mask for val in df_new['id'].values]
    df_new['sector'] = [(val >> 8 + sector_offset) & sector_mask for val in df_new['id'].values]
    df_new = df_new.query('zside == 1 and subdet == 3')

    #patch_dict = dict()
    #for layer in tqdm(df_new.layer.unique(), desc='layers'):
    #    df_layer = df_new.query(f'layer == {layer}')
    #    patch_dict[layer] = dict()
    #    for sector in tqdm(df_new.sector.unique(), desc='sectors'):
    #        df_sector = df_layer.query(f'sector == {sector}')
    #        patch_dict[layer][sector] = dict()
    #        for panel in tqdm(df_new.panel.unique(), desc='panel'):
    #            df_panel            = df_sector.query(f'panel == {panel}')
    #            points              = df_panel[['tc_x', 'tc_y']].values
    #            #ashape, edge_points = hlp.alpha_shape(points, alpha=0.4)

    #            patch_dict[layer][sector][panel] = points #ashape

    layer = 10 
    sector = 1
    df_layer = df_new.query(f'layer == {layer}')
    df_sector = df_layer.query(f'sector == 1')
    patch_dict = dict()
    #for sector in tqdm(df_new.sector.unique(), desc='sectors'):
    df_sector = df_layer.query(f'sector == {sector}')
    patch_dict[sector] = dict()
    for panel in tqdm(df_new.panel.unique(), desc='panel'):
        df_panel            = df_sector.query(f'panel == {panel}')
        points              = df_panel[['tc_x', 'tc_y']].values
        ashape, edge_points = hlp.alpha_shape(points, alpha=0.4)

        patch_dict[sector][panel] = ashape

    outfile = open(f'data/panels/patches_zplus_{layer}.pkl', 'wb')
    pickle.dump(patch_dict, outfile)


