
import torch
from torch import Dataset

class HGCalTCModuleData(Dataset):
    '''
    Data format for individual wafers using (cellu, cellv) formatting.
    '''
    def __init__(self, input_files, targets, transform=None):
        self.input_files = input_files
        self.transform   = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

    def blacklist(self, npzs):
        """
        Remove a list of npzs from the dataset
        Useful to remove bad events
        """
        for npz in npzs: self.npzs.remove(npz)

    def get(self, i):
        d = np.load(self.npzs[i])
        x = d['recHitFeatures']
        if self.flip and np.mean(x[:,7]) < 0:
            # Negative endcap: Flip z-dependent coordinates
            x[:,1] *= -1 # eta
            x[:,7] *= -1 # z

        cluster_index = incremental_cluster_index_np(d['recHitTruthClusterIdx'].squeeze())
        if np.all(cluster_index == 0):
            print('WARNING: No objects in', self.npzs[i])

        truth_cluster_props = np.hstack((
            d['recHitTruthEnergy'],
            d['recHitTruthPosition'],
            d['recHitTruthTime'],
            d['recHitTruthID'],
            ))
        assert truth_cluster_props.shape == (x.shape[0], 5)
        order = cluster_index.argsort()

        return Data(
            x = torch.from_numpy(x[order]).type(torch.float),
            y = torch.from_numpy(cluster_index[order]).type(torch.int),
            truth_cluster_props = torch.from_numpy(truth_cluster_props[order]).type(torch.float),
            inpz = torch.Tensor([i])
            )

    def __len__(self):
        return len(self.npzs)

    def len(self):
        return len(self.npzs)

    def split(self, fraction):
        """
        Creates two new instances of TauDataset with a fraction of events split
        """
        left = self.__class__(self.root)
        right = self.__class__(self.root)
        split_index = int(fraction*len(self))
        left.npzs = self.npzs[:split_index]
        right.npzs = self.npzs[split_index:]

        return left, right

