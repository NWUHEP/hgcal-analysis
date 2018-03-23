
import numpy as np
import pandas as pd

def threshold_algos(df, sort_by, name, ascending=False, nbx=8):
    df = df.sort_values(sort_by, ascending=ascending)
    df.loc[:,name] = True
    if df.shape[0] > nbx*10:
        df.loc[nbx*10:,name] = False


    return df

def algos_8bx(df):

    # 8bx no sort
    df = df.reset_index(drop=True) # causes problems otherwise
    df = threshold_algos(df, 
                         sort_by=['ievt', 'cell'], 
                         name='threshold_8bx_nosort',
                         ascending=True, 
                         nbx=8
                         )
    # 8bx energy sort
    df = threshold_algos(df, 
                         sort_by='reco_e', 
                         name='threshold_8bx_esort',
                         ascending=False, 
                         nbx=8
                         )
    return df

def algos_1bx(df):

    # 8bx no sort
    df = df.reset_index(drop=True) # causes problems otherwise
    df = threshold_algos(df, 
                         sort_by=['ievt', 'cell'], 
                         name='threshold_1bx_nosort',
                         ascending=True, 
                         nbx=1
                         )
         # 8bx energy sort
    df = threshold_algos(df, 
                         sort_by='reco_e', 
                         name='threshold_1bx_esort',
                         ascending=False, 
                         nbx=1
                         )
    return df
