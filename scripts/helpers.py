
import sys
import numpy as np
#import matplotlib.pyplot as plt
import scipy.constants as consts

sys.path.append('/usr/local/lib')
from root_pandas import read_root


def assign_phi(df):
    x, y = df.x.values, df.y.values
    quad2 = (x <= 0) & (y > 0)
    quad3 = (x <= 0) & (y <= 0)
    quad4 = (x > 0) & (y <= 0)

    df.loc[quad2, 'phi'] = np.pi - df.loc[quad2, 'phi']
    df.loc[quad3, 'phi'] = np.pi + df.loc[quad3, 'phi']
    df.loc[quad4, 'phi'] = 2*np.pi - df.loc[quad4, 'phi']

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

def root_to_dataframe(path, features, index):
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

    df = read_root(path, columns=[index]+features, flatten=True)
    df.index = [df[index], df['__array_index']]
    df = df[features]
    return df
