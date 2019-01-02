
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from root_pandas import read_root
from tqdm import tqdm

import scripts.helpers as hlp

def alpha_shape(points, alpha):
    """
    Taken from http://blog.thehumangeo.com/2014/05/12/drawing-boundaries-in-python/
    Compute the alpha shape (concave hull) of a set of points.

    @param points: Iterable container of points.
    @param alpha: alpha value to influence the
                  gooeyness of the border. Smaller numbers
                  don't fall inward as much as larger numbers.
                  Too large, and you lose everything!
    """

    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add((i, j))
        edge_points.append(coords[[i, j]])

    tri = Delaunay(points) # this guy is a bitch
    edges = set()
    edge_points = []
    coords = points

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the
    # triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        try:
            area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        except ValueError:
            print(a, b, c, s)

        if area == 0: continue
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)

    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))

    return cascaded_union(triangles), edge_points


if __name__ == '__main__':

    panel_offset  = 0
    panel_mask    = 0x1F
    sector_offset = 5
    sector_mask   = 0x7
    features = ['id', 'zside', 'subdet', 'layer', 'module',
                #'x', 'y', 'z',
                'tc_zside', 'tc_layer', 'tc_wafer', 'tc_subdet', 'tc_cell', 
                'tc_x', 'tc_y'
                ]

    df_new = read_root('data/test_triggergeom_newmap.root', 'hgcaltriggergeomtester/TreeModules', columns=features, flatten=features[5:])
    df_new['panel']  = [(val >> 8) & panel_mask for val in df_new['id'].values]
    df_new['sector'] = [(val >> 8 + sector_offset) & sector_mask for val in df_new['id'].values]
    
    cell_map = df_new[['tc_zside', 'tc_layer', 'tc_wafer', 'tc_cell', 'tc_x', 'tc_y']]
    cell_map = cell_map.set_index(['tc_zside', 'tc_layer', 'tc_wafer', 'tc_cell']).sort_index()
    cell_map.to_pickle('data/cell_map.pkl')

    df_new = df_new.query('zside == 1')
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
    #            #ashape, edge_points = alpha_shape(points, alpha=0.4)

    #            patch_dict[layer][sector][panel] = points #ashape

    #layer = 1 
    #sector = 1
    #df_layer = df_new.query(f'layer == {layer}')
    #df_sector = df_layer.query(f'sector == 1')
    #patch_dict = dict()
    ##for sector in tqdm(df_new.sector.unique(), desc='sectors'):
    #df_sector = df_layer.query(f'sector == {sector}')
    #patch_dict[sector] = dict()
    #for panel in tqdm(df_new.panel.unique(), desc='panel'):
    #    df_panel            = df_sector.query(f'panel == {panel}')
    #    points              = df_panel[['tc_x', 'tc_y']].values
    #    ashape, edge_points = alpha_shape(points, alpha=0.4)

    #    patch_dict[sector][panel] = ashape

    #outfile = open(f'data/panels/patches_zplus_{layer}.pkl', 'wb')
    #pickle.dump(patch_dict, outfile)


