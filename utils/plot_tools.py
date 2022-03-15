import os
import pickle
from itertools import product
from glob import glob

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, Polygon
from tqdm.notebook import tqdm

import utils.geometry_tools as gt

#from scipy.optimize import lsq_linear
#from sklearn.linear_model import LinearRegression

matplotlib.rcParams.update({'font.size': 18, 'figure.facecolor':'white', 'figure.figsize':(8, 8)})

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

def draw_hgcal_layer(layer=1, hex_radius=0.95*8*2.54/2, include_index=True):
    '''
    This plots a single layer of HGCal in the x-y plane.

    To-do:
    ======
     * Get actual geometry for each layer (a list of wafer (u,v) tuples should suffice)
     * implement partial hex boards
    '''

    # generate coordinates to be used for hexagonal grid.  
    angle = np.pi/6
    u = np.arange(-10, 11)
    hex_coord = np.array(list(product(u, u))).T
    cart_coord = gt.hex_to_cartesian(hex_coord)

    # plot the full grid of wafers with u,v numbering
    fig, ax = plt.subplots(1, figsize=(16, 16))
    ax.set_aspect('equal')

    # draw hexegonal grid
    inner_radius = 0.82*32.8 # 32.8 comes from the TDR, but it's different than what is in simulation
    outer_radius = 160
    for xy, uv in zip(cart_coord.T, hex_coord.T):
        x, y = xy
        u, v = uv

        # filter out wafers
        r = np.sqrt(x**2 + y**2)
        if r < inner_radius or r > outer_radius:
            continue

        color = 'C0'
        alpha = 0.05
        poly = RegularPolygon((x, y), 
                             numVertices=6,
                             radius=hex_radius,
                             orientation=np.radians(0),
                             facecolor=color,
                             alpha=alpha,
                             edgecolor='k',
                             zorder=1
                            )
        ax.add_patch(poly)

        # Add text labels
        ax.text(x, y+0.2, f'({u}, {v})', ha='center', va='center', size=10)

    # inner hexagon and circle (just for show)
    inner_circle = Circle((0, 0), 
                         radius=inner_radius,
                         facecolor='none', 
                         alpha=0.5, 
                         linestyle='--',
                         edgecolor='r'
                         )
    ax.add_patch(inner_circle)

    poly = RegularPolygon((0, 0), 
                         numVertices=6, 
                         radius=inner_radius/np.cos(angle), 
                         orientation=np.radians(0), 
                         facecolor='none', 
                         alpha=0.5, 
                         linestyle='--',
                         edgecolor='r'
                        )
    ax.add_patch(poly)

    # outer hexagon and circle (just for show)
    outer_circle = Circle((0, 0), 
                         radius=160, 
                         facecolor='none', 
                         alpha=0.5, 
                         linestyle='--',
                         edgecolor='r'
                         )
    ax.add_patch(outer_circle)

    poly = RegularPolygon((0, 0), 
                         numVertices=6, 
                         radius=160, 
                         orientation=np.radians(30), 
                         facecolor='none', 
                         alpha=0.5, 
                         edgecolor='r'
                        )

    # draw lines every pi/3 radians
    for angle in np.arange(0, 360, 60):
        rad = (angle/180)*np.pi
        plt.plot([0., 200*np.cos(rad)], [0., 200*np.sin(rad)], 'r--', linewidth=0.5, alpha=0.5)
        
    ax.set_xlim(-170, 170)
    ax.set_ylim(-170, 170)
                  
    if save_and_show:
        plt.savefig('plots/wafer_uv_mapping.pdf')
        plt.show()
    else:
        return fig, ax

