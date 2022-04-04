import os
import pickle
from itertools import product
from glob import glob

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors
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

def draw_hgcal_layer(ax,
        wafer_data=None,
        layer=1,
        hex_radius=gt.hgcal_hex_radius,
        background_wafers=True,
        single_wedge=False,
        include_index=True,
        ):
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

    # draw hexegonal grid
    inner_radius = 0.82*32.8 # 32.8 comes from the TDR, but it's different than what is in simulation
    outer_radius = 159
    cmap = matplotlib.cm.get_cmap('Reds')
    for xy, uv in zip(cart_coord.T, hex_coord.T):
        x, y = xy
        u, v = uv

        # filter out wafers
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan(y/(x + 0.01))
        if r < inner_radius or r > outer_radius:
            continue

        alpha = 0.3
        if single_wedge and ((phi < -0.05 or phi > 2*angle - 0.05) or (x < 0. or y < 0.)):
            alpha = 0.1
            #continue

        if wafer_data is not None and (u, v) in wafer_data.index:
            color = cmap(wafer_data[(u, v)]/wafer_data.max())
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
        elif background_wafers:
            color = 'C0'
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
        if include_index and alpha > 0.1:
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
        ax.plot([0., 200*np.cos(rad)], [0., 200*np.sin(rad)], 'r--', linewidth=0.5, alpha=0.5)

    ax.set_xlabel('X [cm]')
    ax.set_ylabel('Y [cm]')
    if single_wedge:
        ax.set_xlim(0, 170)
        ax.set_ylim(-20, 140)
    else:
        ax.set_xlim(-170, 170)
        ax.set_ylim(-170, 170)

    return ax

def draw_single_module(ax, 
        uv=(0, 0), 
        cell_data=None,
        hex_radius=gt.hgcal_hex_radius,
        do_fill=False,
        include_tc_index=False
        ):
    '''
    Draws a single HGCal wafer/module including HGROC boundaries.  If cell_data
    is provided this can be used to visualize energy deposits in each trigger
    cell.
    '''

    # calculate offset in cartesian coordinates
    xy_offset = gt.hex_to_cartesian(uv)
    x_offset, y_offset = xy_offset

    # draw hexegonal grid
    poly = RegularPolygon((x_offset, y_offset),
                         numVertices=6,
                         radius=hex_radius,
                         orientation=np.radians(0),
                         facecolor='C0',
                         alpha=0.1,
                         edgecolor='k',
                         zorder=1
                            )
    #ax.add_patch(poly)

    # Draw HGROC boundaries
    angle = np.pi/6
    color_list = ['g', 'b', 'r']
    for orientation in [1, 2, 3]:
        color = color_list[orientation - 1] if do_fill else 'k'
        hgroc_patch = Polygon(xy=gt.get_tc_rhombus(orientation, [x_offset, y_offset]), 
                fill=do_fill, 
                color=color, 
                alpha=0.6
                )
        ax.add_patch(hgroc_patch)

    #ax.set_xlim(x_offset - 9, x_offset + 9)
    #ax.set_ylim(y_offset - 10, y_offset + 10)
    #ax.set_xlabel('X [cm]')
    #ax.set_ylabel('Y [cm]')

    cmap = matplotlib.cm.get_cmap('viridis')
    df_uv_to_xy = pd.read_csv('data/tc_uv_to_xy.csv').set_index(['tc_cellu', 'tc_cellv'])
    for (u, v), (x, y) in df_uv_to_xy.iterrows():
        u, v = int(u), int(v)
        tc_radius = gt.hgcal_hex_radius/4
        x, y = x - x_offset, y - y_offset

        if include_tc_index:
            ax.text(x, y, f'({u}, {v})', ha='center', va='center', size=14, color='w' if do_fill else'k')

        orientation = gt.wafer_mask_hgroc[u, v]
        hgroc_angle = np.pi*(1/6 + 2*orientation/3 + 0.063)  
        x, y = x - 0.5*tc_radius*np.cos(hgroc_angle), y - 0.5*tc_radius*np.sin(hgroc_angle)
        tc_xy = gt.get_tc_rhombus(orientation, [x, y], hex_radius=tc_radius)
        if cell_data is not None:
            color = cmap(cell_data.loc[u, v]/cell_data.max())
            tc_patch = Polygon(xy=tc_xy, 
                    fill=True, 
                    edgecolor='k',
                    facecolor=color, 
                    linewidth=4.,
                    alpha=1.,
                    zorder=2,
                    )
        else:
            color = color_list[orientation - 1]
            tc_patch = Polygon(xy=tc_xy, 
                    fill=False, 
                    color=color, 
                    linewidth=3.,
                    alpha=1.,
                    zorder=2
                    )
        ax.add_patch(tc_patch)

    return ax

def animate_wafer_neighbor_shower(ax, df_event,):
    pass

