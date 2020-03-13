import torch
from torch import nn
import numpy as np
import pdb

class EEG_CNN(nn.Module):
    def __init__(self, in_channels, seq_len, out_channels, cnnfilter_size, cnnfilter_stride, cnn_pad, use_bias, pool_size, pool_stride, num_classes, use_batchnorm, eps=1e-5, momentum=0.9, affine=False, dropout = 0, dropoutFC = 0):
        """
        in_channels = number of input features (like color channels in RBG images for example)
        seq_len = length of our temporal data
        out_channels = number of convolutional filters to use
        cnnfilter_size = kernel size for CNN
        cnnfilter_stride = stride of kernel
        cnn_padding = padding to use for data
        use_bias = whether CNN layer has a bias term
        pool_size = MAX pooling kernel size
        pool_stride = MAX pooling stride for kernel
        num_classes = number of output classes for classification task
        use_batchnorm = use batchnorm if True
        eps = batchnorm epsion
        momentum = batchnorm momentum
        affine = batchnorm affine (True or False)
        dropout = if nonzero, probability of neuron dropping out
        """
        
        
        super(EEG_CNN, self).__init__()
        
        #useful params to keep
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.CNN_bias = use_bias
        self.dropoutFC = dropoutFC
        #CNN Layer
        self.CNN = nn.Conv1d(in_channels, out_channels, cnnfilter_size, stride=cnnfilter_stride, padding=cnn_pad, bias=use_bias, padding_mode='zeros')
        Lout = (seq_len + 2*self.CNN.padding[0] - self.CNN.dilation[0]*(self.CNN.kernel_size[0]-1)-1)/self.CNN.stride[0] +1
        
        #initialize the CNN weights with Xavier Norm Init
        nn.init.xavier_normal_(self.CNN.weight.data)
        if self.CNN_bias:
            nn.init.xavier_normal_(self.CNN.bias.data)   
        
        #RELU Layer
        self.RELU = nn.ReLU()
        
        #Batchnorm Layer
        if use_batchnorm:
            self.BatchNorm = nn.BatchNorm1d(out_channels, eps, momentum, affine)
        
        #Dropout Layer
        if dropout > 0:
            self.DropOut = nn.Dropout(dropout)
        
        #MaxPool Layer
        self.MaxPool = nn.MaxPool1d(pool_size, pool_stride)
        
        Lout = int((Lout + 2*self.MaxPool.padding - self.MaxPool.dilation*(self.MaxPool.kernel_size -1) -1)/self.MaxPool.stride +1)
        self.MaxPool_Out = Lout     
        
        #Fully Connected Layers
        self.FC1 = nn.Linear(Lout*out_channels, Lout)
        
        #Dropout in between
        if self.dropoutFC > 0:
            self.DropOutFC = nn.Dropout(dropoutFC)
            
        self.FC2 = nn.Linear(Lout, num_classes)
        
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


        out = self.MaxPool(out)  
        #print(f"Size of MaxPool output {out.size()}")

        #Flatten Layer
        out = out.reshape(out.size(0), self.CNN.out_channels*self.MaxPool_Out)
        #print(f"Size of Flattened output {out.size()}")

        out = self.FC1(out)
        #print(f"Size of FC1 output {out.size()}")
        
        if self.dropoutFC > 0:
            out = self.DropOutFC(out)
        
        out = self.FC2(out)
        #print(f"Size of FC2 output {out.size()}")
            
        return out
        
        
