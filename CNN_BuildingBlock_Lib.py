import torch
from torch import nn
import numpy as np
import pdb

class EEG_CNN_BuildingBlock(nn.Module):
    def __init__(self, in_channels, seq_len, out_channels, cnnfilter_size, cnnfilter_stride, cnn_pad, use_bias, use_maxpool, pool_size, pool_stride, use_batchnorm, eps=1e-5, momentum=0.9, affine=False, dropout = 0):
        """
        For Building Repeatable Blocks. Just CNN_1d, ReLU, Opt. Batchnorm, Opt. Dropout, Opt. MaxPool
        
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
        
        dropout = if nonzero, probability of neuron dropping out
        """
        
        
        super(EEG_CNN_BuildingBlock, self).__init__()
        
        #useful params to keep
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.CNN_bias = use_bias
        self.use_maxpool = use_maxpool
                
        #CNN Layer
        if cnn_pad == 'same': #for keeping dimensionality of input/output the same
            cnn_pad = int(((cnnfilter_stride-1)*seq_len - cnnfilter_stride + cnnfilter_size) // 2)
        
        self.CNN = nn.Conv1d(in_channels, out_channels, cnnfilter_size, stride=cnnfilter_stride, padding=cnn_pad, bias=use_bias, padding_mode='zeros')
        
        #for use in chaining to another model
        self.Lout = int((seq_len + 2*self.CNN.padding[0] - self.CNN.dilation[0]*(self.CNN.kernel_size[0]-1)-1)/self.CNN.stride[0] +1)
        
            
        #initialize the CNN weights with Xavier Norm Init
        nn.init.xavier_normal_(self.CNN.weight.data)
        if self.CNN_bias:
            self.CNN.bias.data.uniform_(0, 1)   
        
        #RELU Layer
        self.RELU = nn.ReLU()
        
        #Batchnorm Layer
        if use_batchnorm:
            self.BatchNorm = nn.BatchNorm1d(out_channels, eps, momentum, affine)
        
        #Dropout Layer
        if dropout > 0:
            self.DropOut = nn.Dropout(dropout)
        
        #MaxPool Layer
        if use_maxpool:
            self.MaxPool = nn.MaxPool1d(pool_size, pool_stride) 
            if type(cnn_pad) is int:
                #for use in chaining to another model.
                self.Lout = int((self.Lout + 2*self.MaxPool.padding - self.MaxPool.dilation*(self.MaxPool.kernel_size -1) -1)/self.MaxPool.stride +1)
            else:
                self.Lout = int(seq_len)
              
        self.out_channels = self.CNN.out_channels
        
    def forward(self, x):
        out = self.CNN(x)
        #print(f"Size of CNN output {out.size()}")
        out = self.RELU(out)
        #print(f"Size of ReLU output {out.size()}")
        
        if self.use_batchnorm:
            out = self.BatchNorm(out)
            #print(f"Size of BatchNorm output {out.size()}")

        if self.dropout > 0:
            out = self.DropOut(out)
            #print(f"Size of DropOut output {out.size()}")

        if self.use_maxpool:
            out = self.MaxPool(out)  
            #print(f"Size of MaxPool output {out.size()}")
            
        return out
    
    
 #The BATCHNORM LAYER HAPPENS BEFORE THE RELU
class EEG_CNN_BuildingBlock2(nn.Module):
    def __init__(self, in_channels, seq_len, out_channels, cnnfilter_size, cnnfilter_stride, cnn_pad, use_bias, use_maxpool, pool_size, pool_stride, use_batchnorm, eps=1e-5, momentum=0.9, affine=False, dropout = 0):
        """
        For Building Repeatable Blocks. Just CNN_1d, ReLU, Opt. Batchnorm, Opt. Dropout, Opt. MaxPool
        
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
        
        dropout = if nonzero, probability of neuron dropping out
        """
        
        
        super(EEG_CNN_BuildingBlock2, self).__init__()
        
        #useful params to keep
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.CNN_bias = use_bias
        self.use_maxpool = use_maxpool
                
        #CNN Layer
        if cnn_pad == 'same': #for keeping dimensionality of input/output the same
            cnn_pad = int(((cnnfilter_stride-1)*seq_len - cnnfilter_stride + cnnfilter_size) // 2)
        
        self.CNN = nn.Conv1d(in_channels, out_channels, cnnfilter_size, stride=cnnfilter_stride, padding=cnn_pad, bias=use_bias, padding_mode='zeros')
        
        #for use in chaining to another model
        self.Lout = int((seq_len + 2*self.CNN.padding[0] - self.CNN.dilation[0]*(self.CNN.kernel_size[0]-1)-1)/self.CNN.stride[0] +1)
        
            
        #initialize the CNN weights with Xavier Norm Init
        nn.init.xavier_normal_(self.CNN.weight.data)
        if self.CNN_bias:
            self.CNN.bias.data.uniform_(0, 1)   
     
        #Batchnorm Layer
        if use_batchnorm:
            self.BatchNorm = nn.BatchNorm1d(out_channels, eps, momentum, affine)
        
        #RELU Layer
        self.RELU = nn.ReLU()
        
        #Dropout Layer
        if dropout > 0:
            self.DropOut = nn.Dropout(dropout)
        
        #MaxPool Layer
        if use_maxpool:
            self.MaxPool = nn.MaxPool1d(pool_size, pool_stride) 
            if type(cnn_pad) is int:
                #for use in chaining to another model.
                self.Lout = int((self.Lout + 2*self.MaxPool.padding - self.MaxPool.dilation*(self.MaxPool.kernel_size -1) -1)/self.MaxPool.stride +1)
            else:
                self.Lout = int(seq_len)
              
        self.out_channels = self.CNN.out_channels
        
    def forward(self, x):
        out = self.CNN(x)
        #print(f"Size of CNN output {out.size()}")        
        if self.use_batchnorm:
            out = self.BatchNorm(out)
            #print(f"Size of BatchNorm output {out.size()}")
        
        out = self.RELU(out)
        #print(f"Size of ReLU output {out.size()}")
        
        if self.dropout > 0:
            out = self.DropOut(out)
            #print(f"Size of DropOut output {out.size()}")

        if self.use_maxpool:
            out = self.MaxPool(out)  
            #print(f"Size of MaxPool output {out.size()}")
            
        return out
        
        
