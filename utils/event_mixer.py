
import os
import pickle
from glob import glob

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

import utils.geometry_tools as gt
import utils.plot_tools as pt


class EventMixer():
    '''
    Offline event mixing.
    '''
    def __init__(self, filenames=None, cuts=None):
        self._filenames = filenames
        self._cuts = 'tc_zside == -1 and tc_subdet == 1  and tc_energy > 0.1'

    def get_events_in_neighborhood(self, uv):
        '''
        Collect all events in a single neighborhood for generating mixed events.
        '''

        list_df_tc = []
        list_df_gen = []
        for filename in tqdm(self._filenames):
            # get the data
            input_file = open(filename, 'rb')
            data_dict = pickle.load(input_file)
            df_tc = data_dict['tc']
            df_gen = data_dict['gen'].query('genpart_exeta < 0.')
        
            # apply some cuts
            df_cut = df_tc.query(self._cuts)
        
            # get events in neighborhood
            events = gt.get_events_in_neighborhood(uv, df_cut)
            
            list_df_tc.append(df_cut.loc[events])
            list_df_gen.append(df_gen.loc[events])
            # implement this!
            
        df_events = pd.concat(list_df_tc)
        df_gen = pd.concat(list_df_gen)
        return df_events, df_gen

    def get_event_mixtures(self, n_events, uv, single_wafer=False):
        '''
        Produces a list of events that mix a number of events in a hexagonal neighborhood defined by uv.
        '''
        df_events, df_gen = self.get_events_in_neighborhood(uv)
        event_ix = df_events.index.unique(level=0)
        event_mixtures_series = []
        event_mixtures = []
        wafer_uv = ['tc_layer', 'tc_waferu', 'tc_waferv', 'tc_cellu', 'tc_cellv']
        for i in tqdm(range(n_events)):
            
            # get mixtures
            size = np.max([np.random.poisson(10), 1])
            events = np.random.choice(event_ix, size=size, replace=False)
            df_mix = df_events.loc[events]
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

        return event_mixtures


if __name__ == '__main__':
    filenames = glob('local_data/tc_data/*')
    event_mixer = EventMixer(filenames)
    event_mixtures = event_mixer.get_event_mixtures(10000, (5, 2))

    outfile = open('local_data/mixtures/single_photon_5_2_test.pkl')
    pickle.dump(event_mixtures, outfile)
    outfile.close()
