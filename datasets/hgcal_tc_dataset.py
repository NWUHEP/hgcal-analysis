
import pickle # switch to something else (npz or hdf5 probably)
import numpy as np
import torch
from torch.utils.data import Dataset

import utils.geometry_tools as gt

class HGCalTCModuleDictDataset(Dataset):
    '''
    Data format for individual wafers using (cellu, cellv) formatting.  This is
    the first prototype of the econ dataset and will not include layer of wafer
    location information.
    '''
    def __init__(self, input_files, do_stacks=False, targets=None, transform=None):
        self.input_files = input_files

        # add something meaningful for these
        self.targets     = targets
        self.transform   = transform

        # unpack the data into 8x8 tensors
        self.wafer_data = self.unpack_wafer_data(do_stacks)

    def __getitem__(self, index):
        #wafer, uv = self.wafer_data[index]
        wafer_data = self.wafer_data[index]
        #y = self.targets[index]

        if self.transform:
            wafer = self.transform(wafer)

        #return wafer, uv
        return wafer_data

    def __len__(self):
        return len(self.wafer_data)

    def unpack_wafer_data(self, do_stacks):
        wafers = []
        for filename in self.input_files:
            f = open(filename, 'rb')
            data = pickle.load(f)
            for (event, zside), event_data in data.items():
                for uv, tc_stack in event_data.items():
                    #layer_index = list(range(len(gt.layer_bins)))
                    uv, _ = gt.map_to_first_wedge(uv)
                    u_bin = np.digitize(uv[0], bins=gt.wafer_bins)
                    wafers.append((torch.tensor(tc_stack).float(), u_bin))
            break

        # temporary: stack all wafers and remove empty wafers (those should be trained on though)
        return wafers

class HGCalTCModuleDataset(Dataset):
    '''
    Data format for individual wafers that ignores location indices.  This is
    the first prototype of the econ dataset and will not include layer of wafer
    location information.
    '''
    def __init__(self, input_files, do_stacks=False, targets=None, transform=None):
        self.input_files = input_files

        # add something meaningful for these
        self.targets     = targets
        self.transform   = transform

        # unpack the data into 8x8 tensors
        self.wafer_data = self.unpack_wafer_data(do_stacks)

    def __getitem__(self, index):
        wafer, uv = self.wafer_data[index]
        #y = self.targets[index]

        if self.transform:
            wafer = self.transform(wafer)

        return wafer, uv

    def __len__(self):
        return len(self.wafer_data)

    def unpack_wafer_data(self, do_stacks):
        wafers = []
        wafer_uv = []
        for filename in self.input_files:
            f = open(filename, 'rb')
            data = pickle.load(f)
            for (event, zside), event_data in data.items():
                for (waferu, waferv), tc_stack in event_data.items():
                    wafer_uv.append((waferu, waferv))
                    if do_stacks:
                        wafers.append(torch.tensor(tc_stack).float(), toch.tensor([waferu, waferv]))
                    else:
                        wafers.append(tc_stack, toch.tensor([waferu, waferv]))

        # temporary: stack all wafers and remove empty wafers (those should be trained on though)
        if do_stacks:
            return wafers
        else:
            wafers = np.vstack(wafers)
            wafers_sums = wafers.sum(axis=(1, 2))
            wafers = wafers[wafers_sums > 10]
            return torch.tensor(wafers).float()
