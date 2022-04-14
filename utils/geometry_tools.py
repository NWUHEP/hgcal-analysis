'''
Data and tools for handling HGCal trigger cell geometry for training ML models.
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from itertools import product

# hex to array masks and mapping
hgcal_hex_radius = 0.95*8*2.54/2
conv_mask = np.array([
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1]
    ])

# useful for coloring and orienting trigger cells
wafer_mask_hgroc = np.array([
    [2, 2, 2, 2, 0, 0, 0, 0],
    [3, 2, 2, 2, 2, 0, 0, 0],
    [3, 3, 2, 2, 2, 2, 0, 0],
    [3, 3, 3, 2, 2, 2, 2, 0],
    [3, 3, 3, 3, 1, 1, 1, 1],
    [0, 3, 3, 3, 1, 1, 1, 1],
    [0, 0, 3, 3, 1, 1, 1, 1],
    [0, 0, 0, 3, 1, 1, 1, 1],
    ])

# used for masking out non-physical entries when mapping a single hex module to
# a grid
wafer_mask_8x8 = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1],
    ])
wafer_mask_14x8x8 = np.tile(wafer_mask_8x8, (14, 1)).reshape(14, 8, 8)

# for mapping into a hex neighborhood grid
wafer_uv_offsets = {
        (0, 0)   : (8, 8),
        (1, 0)   : (4, 0),
        (0, 1)   : (16, 12),
        (1, 1)   : (12, 4),
        (-1, -1) : (4, 12),
        (-1, 0)  : (12, 16),
        (0, -1)  : (0, 4),
       }

# mask for full neighborhood in first 14 EE layers
wafer_mask_14x24x24 = np.zeros((14, 24, 24))
for uv, offset in wafer_uv_offsets.items():
    wafer_mask_14x24x24[:, offset[0]:offset[0] + 8, offset[1]:offset[1] + 8] += wafer_mask_14x8x8

# binning of wafer coordinates to unique encoder modules
wafer_bins = [2, 4, 7, 11]
layer_bins = [4, 8, 10, 12, 16, 20, 28]

# patch definitions for HGROC and trigger cells
def get_tc_rhombus(orientation, xy_offset=[0., 0.], angle=np.pi/6, hex_radius=hgcal_hex_radius):
    '''
    Returns vertex coordinates of rhombus that can be used as a patch for
    visualizing an HGROC or individual trigger cells.
    '''
    if orientation == 1:
        x = np.array([0., -np.cos(angle), -np.cos(angle), 0.])
        y = np.array([0., -np.sin(angle), np.sin(angle),  1.])
    elif orientation == 2:
        x = np.array([0., np.cos(angle),  0.,  -np.cos(angle)])
        y = np.array([0., -np.sin(angle), -1., -np.sin(angle)])
    else:
        x = np.array([0., 0,  np.cos(angle), np.cos(angle)])
        y = np.array([0., 1., np.sin(angle), -np.sin(angle)])

    x = xy_offset[0] + hex_radius*x
    y = xy_offset[1] + hex_radius*y
    return np.vstack([x, y]).T

# helper functions
def hex_neighbors(uv):
    '''
    Produces coordinates of neighboring wafers or trigger cells.
    '''

    u, v = uv
    neighbors = [[u, v + 1], [u + 1, v + 1], 
                 [u - 1, v], [u, v], [u + 1, v], 
                 [u - 1, v - 1], [u, v - 1],  
                 ]
    return neighbors

def delta_phi(phi1, phi2, phi_range=(-np.pi, np.pi)):
    dphi = np.abs(phi2 - phi1)
    if dphi > np.pi:
        dphi = 2*np.pi - dphi

    return dphi

def assign_phi(df):
    x, y = df.x.values, df.y.values
    quad2 = (x <= 0) & (y > 0)
    quad3 = (x <= 0) & (y <= 0)
    quad4 = (x > 0) & (y <= 0)

    df.loc[quad3, 'phi'] = np.pi + df.loc[quad3, 'phi']
    df.loc[quad4, 'phi'] = 2*np.pi - df.loc[quad4, 'phi']
    df.loc[quad2, 'phi'] = np.pi - df.loc[quad2, 'phi']

def eta_to_theta(eta):
    return 2*np.arctan(np.exp(-eta))

def propagate_to_face(theta, phi, pt, z, mass):

    vx = pt*np.cos(phi)*consts.c/mass
    vy = pt*np.sin(phi)*consts.c/mass
    vz = pt/np.tan(theta)*consts.c/mass

    t = z/vz
    x = vx*t
    y = vy*t

    return (x, y, z, t)

def associate_gen_to_cluster(gen, df_cl, by='energy'):
    '''
    Associates gen particles to clusters.  This is done by two methods:

        * 'energy' (default): in this case all clusters in a conde of dR=0.2
        about the gen particles direction are ranked according to pt and the
        highest is selected.  If none are found, no association is made.
        * 'proximity': the cluster that is the closest in dR is selected.
    '''

    # calculate dr
    dphi               = np.abs(gen.phi - df_cl.cl_phi)
    dphi[dphi > np.pi] = 2*np.pi - dphi[dphi > np.pi]
    deta               = np.abs(gen.eta - df_cl.cl_eta)
    dr                 = np.sqrt(deta**2 + dphi**2)

    if by == 'energy':
        df_cl_cut = df_cl[dr < 0.3]
        if df_cl_cut.shape[0] == 0:
            return None
        else:
            return df_cl_cut.sort_values(by='cl_pt', ascending=False).iloc[0]
    else:
        return None


def hex_to_cartesian(hex_coord, angle = np.pi/6, hex_radius=0.95*8*2.54/2, zside=-1):
    '''
    Convert hex coordinates to cartesian coords this should change depending
    on whether you are looking at the +/- z side.  Currently only works of -z
    side.
    '''
    d = 2*hex_radius*np.cos(angle)
    trans_matrix = np.array([[1., -np.sin(angle)], [0., np.cos(angle)]])
    xy = d*np.dot(trans_matrix, hex_coord)

    return xy

def cartesian_rotation_2d(xy_coord, phi):
    '''
    Rotations in two-dimensional using cartesian coordinates.
    '''
    xy_rotation_matrix = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]) 
    xy_rot = np.dot(xy_rotation_matrix, xy_coord.T)
    return xy_rot


def hex_rotation(hex_coord, n):
    '''
    Rotates input coordinates to a new set of coordinates n*pi/3.
    '''
    rho = n*np.pi/3.
    uv_rotation_matrix = np.rint([
        [np.cos(rho) + np.sin(rho)/np.sqrt(3), -2*np.sin(rho)/np.sqrt(3)],
        [2*np.sin(rho)/np.sqrt(3)            , np.cos(rho) - np.sin(rho)/np.sqrt(3)]
       ])
    uv_rot = np.dot(uv_rotation_matrix, np.array(hex_coord)).astype(int)
    return uv_rot

def hex_to_rphi(hex_coord, angle = np.pi/6, hex_radius=0.95*8*2.54/2, zside=-1):
    '''
    Convert hex coordinates to cartesian coords this should change depending
    on whether you are looking at the +/- z side.  Currently only works of -z
    side.
    '''
    x, y = hex_to_cartesian(hex_coord, angle, hex_radius, zside)
    r = np.sqrt(x**2 + y**2)
    phi = np.arcsin(y/r)
    if x < 0.:
        phi = np.pi - phi
    elif y < 0.:
        phi = 2*np.pi + phi

    return r, phi

def map_to_first_wedge(uv, wafer_data=None, angle=np.pi/6, hex_radius=0.95*8*2.54/2, zside=-1):
    '''
    Given a module's (u, v) coordinates, returns the (u', v') coordinates after
    rotating the wafer into the first wedge (modules with phi in [0, pi/3] radians)
    and the integer multiple specifying the total rotation.
    If wafer_data is provided, it will rotate the (u, v) coordinates of all
    entries by the same angle.
    '''

    # first determine the phi coordinate and find
    r, phi = hex_to_rphi(uv)
    iphi = np.floor(phi/(np.pi/3))

    if phi > np.pi/3 + 0.2:
        uv_rot = hex_rotation(uv, -iphi)
    else:
        uv_rot = uv
        iphi = 0

    if wafer_data is not None:
        wafer_data_rot = wafer_data.reset_index()
        uv = wafer_data_rot[['tc_waferu', 'tc_waferv']].values
        uv_rot = hex_rotation(uv.T, -iphi)
        wafer_data_rot['tc_waferu'] = uv_rot[0]
        wafer_data_rot['tc_waferv'] = uv_rot[1]
        wafer_data_rot.set_index(['tc_waferu', 'tc_waferv'], inplace=True)

        return wafer_data_rot['tc_energy'], iphi

    return uv_rot, iphi


def get_events_in_neighborhood(uv, df_data):
    '''
    Filters dataframe down to events with maximum energy in wafer uv.
    '''

    hex_neighborhood = hex_neighbors(uv)
    wafer_group = df_data.groupby(['event', 'tc_waferu', 'tc_waferv'])
    energy_max_idx = wafer_group.sum()[['tc_energy']] .groupby(level=0).idxmax()['tc_energy'].to_list()
    wafer_emap = pd.DataFrame(energy_max_idx, columns=['event', 'waferu', 'waferv']).set_index(['waferu', 'waferv'])
    wafer_mask = wafer_emap.index.isin(hex_neighborhood)
    events = wafer_emap[wafer_mask]['event'].values

    return events


def convert_wafer_to_array(s_tc, single_layer=True):
    '''
    Takes a series of trigger cells indexed by (layer, cellu, cellv) and
    converts all entries single wafer into an 8 by 8 grid.
    '''

    if single_layer:
        wafer_grid = np.zeros((8, 8))
        for (cellu, cellv), e in s_tc.items():
            wafer_grid[cellu, cellv] = e
    else:
        wafer_grid = np.zeros((14, 8, 8))
        for (layer, cellu, cellv), e in s_tc.items():
            layer = int((layer - 1)/2)
            wafer_grid[layer, cellu, cellv] = e
    
    return wafer_grid


def convert_wafer_neighborhood_to_array(s_tc, hex_uv, single_layer=False):
    '''
    Takes a dataframe of trigger cells and converts all entries in the
    neighborhood centered on hex_uv into a 14x24x24 grid.  By default s_tc
    should be indexed as (layer, wafer_u, wafer_v, cell_u, cell_v), but
    optionally a single layer can be processed.  In that case the layer index
    should not be present and the returned grid will be 24x24.

    parameters:
    :s_tc: [pandas.Series]
    '''

    hex_neighborhood = hex_neighbors(hex_uv)
    if single_layer:
        wafer_grid = np.zeros((24, 24))
        wafer_uv = [t[:2] for t in s_tc.index]
    else:
        wafer_grid = np.zeros((14, 24, 24))
        wafer_uv = [t[1:3] for t in s_tc.index]

    for uv in hex_neighborhood:
        relative_uv = uv - np.array(hex_uv)
        uv_offset = wafer_uv_offsets[tuple(relative_uv)]

        if single_layer:
            if tuple(uv) in wafer_uv:
                wafer_data = s_tc.loc[uv[0], uv[1]]
                wafer_array = convert_wafer_to_array(wafer_data, single_layer=single_layer)
                wafer_grid[uv_offset[0]:uv_offset[0] + 8, uv_offset[1]:uv_offset[1] + 8] += wafer_array
        else:
            if tuple(uv) in wafer_uv:
                wafer_data = s_tc.loc[:, uv[0], uv[1]]
                wafer_array = convert_wafer_to_array(wafer_data, single_layer=False)
                wafer_grid[:, uv_offset[0]:uv_offset[0] + 8, uv_offset[1]:uv_offset[1] + 8] += wafer_array

    return wafer_grid
