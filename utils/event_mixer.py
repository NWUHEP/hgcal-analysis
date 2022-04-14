
import os
import pickle
from pathlib import Path
from glob import glob

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

import utils.geometry_tools as gt
import utils.plot_tools as pt


class EventMixer():
    '''
    Offline event mixing.

    parameters
    :filenames: list of filenames containing event dataframes
    :n_avg: this is the average number of events to be included in the neighborhood
    :cuts: additional criteria to filter out events
    '''
    def __init__(self, filenames, n_avg=5, cuts=None):
        self._filenames = filenames
        self._n_avg = n_avg
        self._cuts = cuts

        self.df_events = self._load_dataframes()

    def _load_dataframes(self):
        '''
        Gets all dataframes in files and concatenates them. (implement gen particle)

        '''

        list_df_tc = []
        #list_df_gen = []
        for filename in tqdm(self._filenames, desc='Opening datafiles and appending to event dataframe...'):
            # get the data
            input_file = open(filename, 'rb')
            data_dict = pickle.load(input_file)
            df_tc = data_dict['tc']
            #df_gen = data_dict['gen'].query('genpart_exeta < 0.')
        
            # apply some cuts
            df_cut = df_tc.query(self._cuts)
        
            list_df_tc.append(df_cut)
            #list_df_gen.append(df_gen.loc[events])
            
        df_events = pd.concat(list_df_tc)
        #df_gen = pd.concat(list_df_gen)
        return df_events#, df_gen

    def get_event_mixtures(self, n_events, uv, single_wafer=False):
        '''
        Produces a list of events that mix a number of events in a hexagonal neighborhood defined by uv.
        '''
        #df_events, df_gen = self.get_events_in_neighborhood(uv)
        events_neighborhood = gt.get_events_in_neighborhood(uv, self.df_events)
        df_neighborhood = self.df_events.loc[events_neighborhood]
        event_mixtures_series = []
        event_mixtures = []
        wafer_uv = ['tc_layer', 'tc_waferu', 'tc_waferv', 'tc_cellu', 'tc_cellv']
        for i in tqdm(range(n_events)):
            
            # get mixtures
            size = np.max([np.random.poisson(self._n_avg), 1])
            events_mix = np.random.choice(events_neighborhood, size=size, replace=False)
            df_mix = df_neighborhood.loc[events_mix]
            s_energy = df_mix.groupby(wafer_uv).sum()['tc_energy']
            event_mixtures_series.append(s_energy)
            
            if single_wafer:
                # get wafer with highest energy across all layers
                wafer_sums = df_mix.groupby(wafer_uv[:3]).sum()['tc_energy']
                uv_max = wafer_sums.idxmax()
                array_mix = gt.convert_wafer_to_array(s_energy.loc[:, uv_max[1], uv_max[2]])
                event_mixtures.append([array_mix, (uv_max[1:])])
            else:
                # convert hex neighborhood to array
                neighbor_mix = gt.convert_wafer_neighborhood_to_array(s_energy, uv)
                event_mixtures.append(neighbor_mix)

        return event_mixtures, object_data

if __name__ == '__main__':

    u, v = (5, 2)
    nevents = int(1e5)
    filenames = glob('local_data/tc_data/*')
    event_mixer = EventMixer(filenames, 5, 'tc_zside == -1 and tc_subdet == 1  and tc_energy > 0.1')
    event_mixtures, object_data = event_mixer.get_event_mixtures(nevents, (u, v))

    outdir = Path(f'local_data/event_mixtures/single_photon')
    outdir.mkdir(parents=True, exist_ok=True) 
    outfile = open(outdir / f'events_{u}_{v}_{nevents}.pkl', 'wb')
    pickle.dump(event_mixtures, outfile)
    outfile.close()
