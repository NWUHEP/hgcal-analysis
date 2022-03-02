
import sys, math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import scipy.constants as consts
from scipy.spatial import ConvexHull, Delaunay

import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from tqdm import tqdm

#sys.path.append('/usr/local/lib')
#from root_pandas import read_root

def set_default_style():                    
    import matplotlib                       
    np.set_printoptions(precision=3)        
    matplotlib.style.use('default')         
    params = {                              
              'axes.facecolor': 'white',    
              'axes.titlesize':'x-large',   
              'axes.labelsize'    : 19,     
              'xtick.labelsize'   : 16,     
              'ytick.labelsize'   : 16,     
              'figure.titlesize'  : 20,     
              'figure.figsize'    : (8, 8), 
              'legend.fontsize'   : 18,     
              'legend.numpoints'  : 1,      
              'font.serif': 'Arial'         
              }                             
    matplotlib.rcParams.update(params)      

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

