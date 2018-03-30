
from functools import partial
import numpy as np
import pandas as pd

def threshold_algo(df, name, n_readout = 10, sort_by='reco_e', ascending=False):
    df = df.sort_values(sort_by, ascending=ascending)
    df = df.reset_index(drop=True)
    df.loc[:n_readout, name] = True

    return df

def algorithm_test_8bx(df):

    # 8 bx algorithms
    df = threshold_algo(df, 'threshold_8bx_nosort',
                        n_readout = 80, 
                        sort_by=['ievt', 'cell'], 
                        ascending=True
                        )
    df = threshold_algo(df, 'threshold_8bx_esort',
                        n_readout = 80, 
                        sort_by='reco_e', 
                        ascending=False
                        )
    return df

def algorithm_test_1bx(df):

    # 8 bx algorithms
    df = threshold_algo(df, 'threshold_1bx_nosort',
                        n_readout = 10, 
                        sort_by='cell', 
                        ascending=True
                        )
    df = threshold_algo(df, 'threshold_1bx_esort',
                        n_readout = 10, 
                        sort_by='reco_e', 
                        ascending=False
                        )
    return df

