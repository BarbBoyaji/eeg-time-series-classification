import torch
from torch import nn
import numpy as np
import CNN_BuildingBlock_Lib as CNN_BB
import RNN_BuildingBlock_Lib as RNN_BB

import CNN_Lib as cnn_lib
import pdb

class CombinationModel_RNN_CNN(nn.Module):
    def __init__(self, in_channels, seq_len, hidden_dim, num_layers, non_linearity, initialization, out_channels, cnnfilter_size, cnnfilter_stride, cnn_pad, use_bias, num_classes, use_maxpool, pool_size, pool_stride, use_batchnorm, eps=1e-5, momentum=0.9, affine=False, dropout = 0):
        """
        Using Repeatable Blocks to create combined model. 
        A) Variation of RNN
        B) Variations of CNN_1d, ReLU, Opt. Batchnorm, Opt. Dropout, Opt. MaxPool. 
        B) Followed by Variation of CNN1d, ReLU, ... , FC, FC.
        
        num_layers = for rnn layers
        hidden_dim = size of hidden dimension in rnn
        
        in_channels = number of input features (like color channels in RBG images for example)
        seq_len = length of our temporal data
        out_channels = number of convolutional filters to use
        cnnfilter_size = kernel size for CNN
        cnnfilter_stride = stride of kernel
        cnn_padding = padding to use for data
        use_bias = whether CNN layer has a bias term
        
        use_maxpool - whether to use MAX Pooling
        pool_size = MAX pooling kernel size
        pool_stride = MAX pooling stride for kernel
        
        use_batchnorm = use batchnorm if True
        eps = batchnorm epsion
        momentum = batchnorm momentum
        affine = batchnorm affine (True or False)
        
        dropout = if nonzero, probability of neuron dropping out (last element is for FC dropout)
        """
        
        
        super(CombinationModel_RNN_CNN, self).__init__()
        
        #building blocks 
        
        
        self.RNN_Block = RNN_BB.EEG_RNN_BuildingBlock(in_channels, hidden_dim, num_layers, non_linearity, initialization, use_cuda=True)
        
        self.CNN_Block1 = CNN_BB.EEG_CNN_BuildingBlock(self.RNN_Block.Hall, seq_len, out_channels[0], cnnfilter_size[0], cnnfilter_stride[0], cnn_pad[0], use_bias[0], use_maxpool[0], pool_size[0], pool_stride[0], use_batchnorm[0], eps[0], momentum[0], affine[0], dropout[0])
        
        
        self.CNN_Block2 = CNN_BB.EEG_CNN_BuildingBlock(self.CNN_Block1.out_channels, self.CNN_Block1.Lout, out_channels[1], cnnfilter_size[1], cnnfilter_stride[1], cnn_pad[1], use_bias[1], use_maxpool[1], pool_size[1], pool_stride[1], use_batchnorm[1], eps[1], momentum[1], affine[1], dropout[1] )
        
        self.CNN_Block3 = cnn_lib.EEG_CNN(self.CNN_Block2.out_channels, self.CNN_Block2.Lout, out_channels[2], cnnfilter_size[2], cnnfilter_stride[2], cnn_pad[2], use_bias[2], pool_size[2], pool_stride[2], num_classes, use_batchnorm[2], eps[2], momentum[2], affine[2], dropout[2], dropout[3])
        
                
    def forward(self, x):
        out, hidden = self.RNN_Block.forward(x)
        out = out.transpose(2,1)
        out = self.CNN_Block1.forward(out)
        out = self.CNN_Block2.forward(out)
        out = self.CNN_Block3.forward(out)
        
        return out
