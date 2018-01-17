#!/usr/bin/env python

import numpy as np
import pandas as pd
import ROOT as r

if __name__ == '__main__':

    path = '/eos/uscms/store/user/jbueghly/hgcal/'

    # use root to get the event lists without burning memory
    print('getting event lists')
    sig_file = r.TFile(path+'ntuple_singleGamma_1000.root')
    bkg_file = r.TFile(path+'ntuple_doublenu_1000.root')
    sig_tree = sig_file.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    bkg_tree = bkg_file.Get('hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    sig_evts = [e.event for e in sig_tree]
    bkg_evts = [e.event for e in bkg_tree]
    sig_file.Close()
    bkg_file.Close()
    df_out = pd.DataFrame()
    df_out.loc[:, 'sig_evts'] = sig_evts
    df_out.loc[:, 'bkg_evts'] = bkg_evts
    df_out.to_pickle('data/flattuples/evts.pkl')
