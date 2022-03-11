import torch
from torch import nn
import torch.nn.functional as F
from utils.geometry_tools import wafer_mask, conv_mask


class AutoEncoderWafer(nn.Module):
    '''
    Autoencoder for a single HGCal wafer with the following 8x8 encoding:

    1111----
    21111---
    221111--
    2221111-
    22223333
    -2223333
    --223333
    ---23333

    Where 1, 2, 3 indicate entries from one of the three HGCROC "faces".
    Need to write custom 3x3 convolutional kernels that properly encoded
    nearest neigbors:

    ab-
    cde
    -fg

    '''
    def __init__(self, output_dim, device):
        super(AutoEncoderWafer, self).__init__()

        # reusable layers
        self.pool = nn.MaxPool2d(2)

        # encoding layers
        self.conv2d_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1, stride=1, bias=False)
        #self.conv2d_2 = nn.Conv2d(8, 1, kernel_size=2, padding=0, stride=1, bias=True)

        #self.bnorm_1  = nn.BatchNorm2d(num_features=8)
        #self.conv2d_2 = nn.Conv2d(8, 8, kernel_size=2, padding=0, stride=2)
        #self.act_2    = nn.ReLU()
        #self.pool_2   = nn.MaxPool2d(2)
        #self.encode_dense = nn.Sequential(
        #    nn.Linear(8 * 8 * 8, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 32),
        #    nn.ReLU(),
        #    )

        # decoding layers
        #self.decode_dense = nn.Sequential(
        #    nn.Linear(32, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, output_dim),
        #    nn.ReLU()
        #)
        #self.t_conv2d_1 = nn.ConvTranspose2d(1, 8, kernel_size=2, padding=0, stride=2)
        self.t_conv2d_2 = nn.ConvTranspose2d(8, 1, kernel_size=3, padding=2, stride=1, output_padding=0)

        # create mask for hexagonal convolutions
        conv_mask = torch.ones(8, 1, 3, 3)
        conv_mask[:, :, 2, 0] = 0
        conv_mask[:, :, 0, 2] = 0
        self.register_buffer('weight_update_mask', conv_mask.bool())
        self.register_buffer('fixed_weights', torch.zeros(8, 1, 3, 3))

        #self.conv2d_1.weight = nn.Parameter(conv_mask)
        #self.conv_mask  = nn.Parameter(torch.tensor(wafer_mask).view(-1, 3, 3), requires_grad=False)

    def masked_conv2d(self, x, layer, do_inv=False):
        weights = layer.weight
        bias    = layer.bias 

        weights = torch.where(self.weight_update_mask, weights, self.fixed_weights)
        if layer.bias:
            bias = torch.where(self.weight_update_mask, bias, self.fixed_weights)

        if do_inv:
            return F.conv_transpose2d(x, weights, bias, layer.stride, layer.padding)
        else:
            return F.conv2d(x, weights, bias, layer.stride, layer.padding)

    def encode(self, x):
        '''
        Takes image data and returns the encoded values
        '''

        # adds channel dimension for the case that there is one channel per image
        x = x.unsqueeze(1) 

        # carry out convolutions with hex masking
        x = self.masked_conv2d(x, self.conv2d_1)
        x = F.relu(x)
        #x = self.pool(x) # reduces x, y by 2
        #x = self.conv2d_2(x)
        #x = F.relu(x)

        #x = self.bnorm_1(x)
        #x = self.conv2d_2(x) # reduces by 2
        #x = self.act_2(x)
        #x = self.pool_2(x)
        #x = x.view(-1, 8 * 8 * 8)
        #x = self.encode_dense(x)

        return x

    def decode(self, h):
        '''
        Takes encoded input (h) and returns decoded output x'
        '''
        #x = self.decode_dense(h)
        #x = x.view(-1, 1, 8, 8)
        #x = self.upsample_1(x)
        #x = self.act_3(x)
        #x = x.view(-1, 8, 8)

        x = self.t_conv2d_1(h)
        x = F.relu(x)
        #x = self.t_conv2d_2(x)
        #x = torch.relu(x)
        x = x.squeeze()

        return x

    def forward(self, x):
        '''
        Carries out the full encoding+decoding to predict x'
        '''
        #self.conv2d_1.weight = nn.Parameter(torch.where(self.weight_update_mask, self.conv2d_1.weight, self.fixed_weights))
        h = self.encode(x)
        x = self.decode(h)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(32, 64, 2)
        )
        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(64, 32, 2),
            #nn.ReLU(),
            nn.ConvTranspose2d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoderToy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act_1    = nn.ReLU()
        self.pool_1   = nn.MaxPool2d(2)
        self.conv2d_2 = nn.Conv2d(8, 8, kernel_size=2, padding=0, stride=2)
        self.act_2    = nn.ReLU()
        self.pool_2   = nn.MaxPool2d(2)
        self.bnorm1 = nn.BatchNorm2d(num_features=8)
        self.encode_dense = nn.Sequential(
            nn.Linear(8 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
            )
        self.decode_dense = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )
        #self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        #self.act_3 = nn.ReLU()

    def encode(self, x):
        '''
        Takes image data and returns the encoded values
        '''
        x = x.unsqueeze(1) # adds channel dimension for the case that there is one channel per image
        x = self.conv2d_1(x) # preserves shape
        x = self.act_1(x)
        x = self.pool_1(x) # reduces x, y by 2
        x = self.bnorm1(x)
        #x = self.conv2d_2(x) # reduces by 2
        #x = self.act_2(x)
        #x = self.pool_2(x)
        x = x.view(-1, 8 * 8 * 8)
        x = self.encode_dense(x)

        return x

    def decode(self, h):
        '''
        Takes encoded input (h) and returns decoded output x'
        '''
        x = self.decode_dense(h)
        #x = x.view(-1, 1, 8, 8)
        #x = self.upsample_1(x)
        #x = self.act_3(x)
        x = x.view(-1, 16, 16)
        return x

    def forward(self, x):
        '''
        Carries out the full encoding+decoding to predict x'
        '''
        h = self.encode(x)
        x = self.decode(h)
        return x
