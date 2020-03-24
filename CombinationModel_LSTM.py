import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import CNN_BuildingBlock_Lib as CNN_BB
import RNN_BuildingBlock_Lib as RNN_BB

import pdb

class CombinationModel_LSTM(nn.Module):
    def __init__(self, in_channels, seq_len, num_classes, batch_size, hidden_dim1, hidden_dim2, num_layers1, num_layers2, 
                dropoutLSTM, dropoutFC, eps=1e-5, momentum=0.9, affine=False):
        """
        Using Repeatable Blocks to create combined model. 
        """
        super(CombinationModel_LSTM, self).__init__()
        
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_layers2 = num_layers2
        self.seq_len = seq_len
        
        #building blocks 
        self.LSTM1 = nn.LSTM(in_channels, hidden_dim1, num_layers1, bidirectional=True)
        self.LSTM2 = nn.LSTM(2*hidden_dim1, hidden_dim2, num_layers2, bidirectional=True)
        
        #Dropout after LSTM
        self.DropOutLSTM = nn.Dropout(dropoutLSTM)
        
        #Fully Connected Layers
        self.FC1 = nn.Linear(seq_len*hidden_dim2*2, in_channels)
        
        #Batchnorm
        self.BN = nn.BatchNorm1d(in_channels, eps, momentum, affine)
        self.Activ = nn.ELU()
        
        #Dropout after FC
        self.DropOutFC = nn.Dropout(dropoutFC)
 
        self.FC2 = nn.Linear(in_channels, num_classes) 
        
    def forward(self, x):
        #first tried with non-overlapping chunks.

        # Initialize hidden state with zeros
        #h0 = torch.zeros(self.in_channels, x.size(0), self.hidden_dim1).requires_grad_()

        # Initialize cell state
        #c0 = torch.zeros(self.in_channels, x.size(0), self.hidden_dim1).requires_grad_()

        outLSTM1, hidden = self.LSTM1.forward(x)
        outLSTM2, hidden = self.LSTM2.forward(outLSTM1)
        outLSTM = self.DropOutLSTM(outLSTM2)

        #Flatten Layer
        outLSTM = outLSTM.reshape(outLSTM.size(0), self.seq_len*self.hidden_dim2*2)
                
        #Linear Layer
        out = self.FC1(outLSTM)
        
        out = self.BN(out)
        
        out = self.DropOutFC(out)
        out = self.FC2(out)
        
        
        return out
