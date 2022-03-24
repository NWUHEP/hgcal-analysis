'''
Some tools for visualizing detector elements.
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

wafer_mask = np.array([
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 1],
    ])

conv_mask = np.array([
    [1, 1, 0],
    [1, 1, 1],
    [0, 1, 1]
    ])

hgcal_hex_radius = 0.95*8*2.54/2

def hex_neighbors(u, v):
    neighbors = [[u + 1, v], [u, v + 1], 
                 [u - 1, v], [u, v - 1], 
                 [u + 1, v + 1], [u - 1, v - 1] 
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



    
