
import sys, math
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import scipy.constants as consts
from scipy.spatial import ConvexHull, Delaunay

import shapely.geometry as geometry
from shapely.ops import cascaded_union, polygonize
from tqdm import tqdm

#sys.path.append('/usr/local/lib')
#from root_pandas import read_root

def train(model, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    criterion    = nn.MSELoss() # mean square error loss
    optimizer    = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs

