import torch
from torch import Dataset

class ToyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data      = torch.Tensor(data)
        self.targets   = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
