#this will be the file we do our RNN in

#classification may mean using the many to one architecture. 
import torch
from torch import nn
import numpy as np

class EEG_RNN_BuildingBlock(nn.Module):
    """
    num_inputs: expected features in input 
    num_layers: number of recurrent layers
    nonlinearity: 'tanh' or 'relu'
    initialization: xavierNorm or xavierUniform
    """
    def __init__(self, num_inputs, hidden_dim, num_layers, non_linearity, initialization, use_cuda=True, num_directions =1):
        super(EEG_RNN_BuildingBlock, self).__init__()
        
        
        #check the initialization type
        if not (initialization in ['xavierNorm', 'xavierUniform']):
            raise Exception('ValueErr', 'Not An Option')
            print('Acceptable Options include: xavierNorm, xavierUniform')
        
        
        #store own vars
        self.inputsNum = num_inputs
        self.layersNum= num_layers
        self.hiddenDim = hidden_dim
        self.init = initialization
        self.nonlinearity = non_linearity
        self.use_cuda = use_cuda
        
        print("DONE INITIALIZING")
        
        self.RNN = nn.RNN(num_inputs, hidden_dim, num_layers, nonlinearity=non_linearity, batch_first=True)
        
        #look at torch documentation for explanation on what these params represent
        self.S = num_layers*num_directions #unidirectional for now, try bidirectional later
        self.Hall = num_directions*hidden_dim
        self.Hout = hidden_dim
        
    def forward(self, x):
        #use Xavier initialization
        if self.init == 'xavierUniform':
            self.hidden_layer0 = torch.nn.init.xavier_uniform_(torch.empty(self.layersNum, x.size(0), self.hiddenDim),  gain=nn.init.calculate_gain(self.nonlinearity)).requires_grad_()
                
        if self.init == 'xavierNorm':
            self.hidden_layer0 = torch.nn.init.xavier_normal_(torch.empty(self.layersNum, x.size(0), self.hiddenDim),  gain=nn.init.calculate_gain(self.nonlinearity)).requires_grad_()
            
        if self.use_cuda:
            out, hidden_layer = self.RNN(x, self.hidden_layer0.to('cuda:0').detach())
            return out, hidden_layer                                                  
        else:
            out, hidden_layer = self.RNN(x, self.hidden_layer0.detach())
            return out, hidden_layer   
                                                          
                                                          
