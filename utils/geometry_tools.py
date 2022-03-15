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

def hex_rotation(hex_coord, n):
    '''
    Rotates input coordinates to a new set of coordinates n*pi/3.
    '''
    uv_rotation_matrix = np.rint([
        [np.cos(rho) + np.sin(rho)/np.sqrt(3), -2*np.sin(rho)/np.sqrt(3)],
        [2*np.sin(rho)/np.sqrt(3)            , np.cos(rho) - np.sin(rho)/np.sqrt(3)]
       ])
    uv_rot = np.dot(uv_rotation_matrix, np.array(hex_coord)).astype(int)
    return uv_rot

