
import torch
from torch.utils.data import Dataset

class HGCalTCModuleDataset(Dataset):
    '''
    Data format for individual wafers using (cellu, cellv) formatting.  This is
    the first prototype of the econ dataset and will not include layer of wafer
    location information.
    '''
    def __init__(self, input_files, targets, transform=None):
        self.input_files = input_files
        self.transform   = transform

        # unpack the data into 8x8 tensors
        self.wafers = unpack_wafers()

    def __getitem__(self, index):
        x = self.data[index]
        #y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x#, y

    def __len__(self):
        return len(self.data)

    def unpack_wafer_data(self):
        wafers = []
        for filename in self.input_files:
            f = open(filename, 'rb')
            data = pickle.load(f)
            for (event, zside), event_data in data.items():
                for (waferu, waferv), tc_stack in event_data.items():
                    wafers.append(tc_stack)

        self.data = np.vstack(wafers)
