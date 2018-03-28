
from functools import partial
import numpy as np
import pandas as pd

def threshold_algo(df, name, scan_vals, n_readout = 10, sort_by='reco_e', ascending=False):
    mask = np.zeros((df.shape[0], scan_vals.size))
    mask[:n_readout] = True

    cols = [f'{name}_{c}' for c in scan_vals]
    df = df.sort_values(sort_by, ascending=ascending)
    df = df.reset_index(drop=True)
    df.loc[:, cols] = np.logical_and(df.loc[:, cols], mask)

    return df

def algorithm_test(df, mippt_scan):

    # 1 bx algorithms
    func = partial(threshold_algo, 
                   name = 'threshold_1bx_nosort', 
                   scan_vals = mippt_scan, 
                   n_readout=10, 
                   sort_by=['ievt', 'cell'], 
                   ascending=True
                   )
    df = df.groupby('ievt').apply(func)
    
    func = partial(threshold_algo, 
                   name = 'threshold_1bx_esort', 
                   scan_vals = mippt_scan, 
                   n_readout=10, 
                   sort_by='reco_e', 
                   ascending=False
                   )
    df = df.groupby('ievt').apply(func)

    # 8 bx algorithms
    df = threshold_algo(df, 'threshold_8bx_nosort',  mippt_scan,
                        n_readout = 80, 
                        sort_by=['ievt', 'cell'], 
                        ascending=True
                        )
    df = threshold_algo(df, 'threshold_8bx_esort', mippt_scan,
                        n_readout = 80, 
                        sort_by='reco_e', 
                        ascending=False
                        )
    return df

