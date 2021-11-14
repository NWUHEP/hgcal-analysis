
import sys, math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import scipy.constants as consts
from scipy.spatial import Delaunay

import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from tqdm import tqdm

#sys.path.append('/usr/local/lib')
#from root_pandas import read_root


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

#def root_to_dataframe(path, features, index):
#    '''
#    Converts a semi-flat root ntuple to a pandas dataframe using root_pandas.
#
#    Parameters:
#    ===========
#    path : location of input ROOT file
#    features : these are the branches to be put into the dataframe.  These
#               should be vectors of basic types which will be flattened.
#    index : the column to use as the first index (the second index will be
#            __array_index that comes out of the root_pandas flattening).
#    '''
#
#    df = read_root(path, columns=[index]+features, flatten=True)
#    df.index = [df[index], df['__array_index']]
#    df = df[features]
#    return df

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

