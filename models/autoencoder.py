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
    def __init__(self):
        super(AutoEncoderWafer, self).__init__()

        # reusable layers
        self.pool = nn.MaxPool2d(2)

        # encoder layers
        #self.conv2d = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2d_enc = nn.Sequential( 
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            #nn.BatchNorm2d(8)
            #nn.Conv2d(8, 1, 3, stride=1, padding=1),
            #nn.ReLU(),
            )
        self.linear_enc = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(),
            #nn.Linear(64, 16),
            #nn.ReLU()
            )

        # decoder layers
        self.linear_dec = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            #nn.Linear(64, 128),
            #nn.ReLU(),
            )
        self.tconv2d_dec = nn.Sequential(
            #nn.ConvTranspose2d(1, 8, 3, stride=1, padding=1),
            #nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        # mask for hexagonal convolutions
        conv_mask = torch.ones(8, 1, 3, 3)
        conv_mask[:, :, 2, 0] = 0
        conv_mask[:, :, 0, 2] = 0
        self.register_buffer('conv2d_weight_update_mask', conv_mask.bool())
        self.register_buffer('conv2d fixed_weights', torch.zeros(8, 1, 3, 3))

        # mask for wafers
        #conv_mask = torch.ones(8, 1, 8, 8)
        #conv_mask[:, :, 0, 4:] = 0
        #conv_mask[:, :, 1, 5:] = 0
        #conv_mask[:, :, 6, 7:] = 0
        #conv_mask[:, :, 7, 8] = 0
        #self.register_buffer('conv2d_weight_update_mask', conv_mask.bool())
        #self.register_buffer('conv2d fixed_weights', torch.zeros(8, 1, 3, 3))

    def masked_conv2d(self, x, layer):
        weights = layer.weight
        bias    = layer.bias 

        weights = torch.where(self.weight_update_mask, weights, self.fixed_weights)
        if layer.bias:
            bias = torch.where(self.weight_update_mask, bias, self.fixed_weights)
        return F.conv2d(x, weights, bias, layer.stride, layer.padding)

    def encode(self, x):
        
        #x = self.masked_conv2d(x, self.conv2d_1)
        #self.conv2d_1.weight = nn.Parameter(torch.where(self.weight_update_mask, self.conv2d_1.weight, self.fixed_weights))
        x = self.conv2d_enc(x)
        x = x.flatten(1)
        x = self.linear_enc(x)
        return x

    def decode(self, x):
        x = self.linear_dec(x)
        x = x.view(-1, 8, 4, 4)
        x = self.tconv2d_dec(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def forward(self, x):
        '''
        Carries out the full encoding + decoding to predict x'
        '''
        #e_total = x.sum(dim=[1, 2, 3])
        h = self.encode(x)
        x = self.decode(h)
        #x *= e_total/x.sum(dim=[1, 2, 3])
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv2d_enc = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),
            #nn.Conv2d(8, 1, 3, stride=1, padding=1),
            #nn.ReLU(),
            )
        self.linear_enc = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU()
            )
        self.linear_dec = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU()
            )
        self.tconv2d_dec = nn.Sequential(
            #nn.ConvTranspose2d(1, 8, 3, stride=1, padding=1),
            #nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def encode(self, x):
        x = self.conv2d_enc(x)
        x = x.flatten(1)
        x = self.linear_enc(x)
        return x

    def decode(self, x):
        x = self.linear_dec(x)
        x = x.view(-1, 8, 4, 4)
        x = self.tconv2d_dec(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
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
        self.bnorm1   =   nn.BatchNorm2d(num_features=8)
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
