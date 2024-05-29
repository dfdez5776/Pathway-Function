import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

######### MULTIREGIONAL MODEL ##########

class RNN_MultiRegional(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim):
        super(RNN_MultiRegional, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                inp_dim: dimension of input
                hid_dim: number of hidden neurons, each region and connection between region has hid_dim/2 neurons
                action_dim: output dimension, should be one for lick or no lick
        '''

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.alm_mask = torch.cat([torch.zeros(size=(int(hid_dim/2),)), torch.ones(size=(int(hid_dim/2),))]).cuda()
        self.str_mask = torch.cat([torch.ones(size=(int(hid_dim/2),)), torch.zeros(size=(int(hid_dim/2),))]).cuda()
        
        # Identity Matrix of 0.5 Not Trained
        self.str2str_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/2), int(hid_dim/2))))
        # Excitatory Connections
        self.str2m1_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/2), int(hid_dim/2))))
        # Mix of Excitatory and Inhibitory Connections
        self.m12m1_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/2), int(hid_dim/2))))
        # Excitatory Connections
        self.m12str_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/2), int(hid_dim/2))))

        nn.init.uniform_(self.str2str_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.str2m1_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.m12m1_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.m12m1_weight_l0_hh, 0, 0.01)

        # Implement Necessary Masks
        # Striatum recurrent weights
        sparse_matrix = torch.empty_like(self.str2str_weight_l0_hh)
        nn.init.sparse_(sparse_matrix, 0.85)
        sparse_mask = torch.where(sparse_matrix != 0, 1, 0).cuda()
        self.str2str_mask = torch.zeros_like(self.str2str_weight_l0_hh).cuda()
        self.str2str_fixed = torch.empty_like(self.str2str_weight_l0_hh).uniform_(0, 0.01).cuda() * sparse_mask
        self.str2str_D = -1*torch.eye(int(hid_dim/2)).cuda()

        self.m12m1_D = torch.eye(int(hid_dim/2)).cuda()
        self.m12m1_D[int(hid_dim/2)-(int( 0.3*(hid_dim/2) )):, 
                        int(hid_dim/2)-(int( 0.3*(hid_dim/2) )):] *= -1
        
        # ALM to striatum weights
        self.m12str_mask_excitatory = torch.ones(size=(int(hid_dim/2), int(hid_dim/2) - int(0.3*(hid_dim/2))))
        self.m12str_mask_inhibitory = torch.zeros(size=(int(hid_dim/2), int(0.3*(hid_dim/2))))
        self.m12str_mask = torch.cat([self.m12str_mask_excitatory, self.m12str_mask_inhibitory], dim=1).cuda()
        
        # Input weights
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim)))
        nn.init.uniform_(self.inp_weight, 0, 0.1)

        # Behavioral output layer
        self.fc1 = nn.Linear(hid_dim, action_dim)

        # Time constants for networks (not sure what would be biologically plausible?)
        t_str = 0.1 * torch.ones(int(hid_dim/2))
        t_m1_excitatory = 0.1 * torch.ones(int(hid_dim/2) - int(0.3*(hid_dim/2)))
        t_m1_inhibitory = 0.1 * torch.ones(int(0.3*(hid_dim/2)))
        self.t_const = torch.cat([t_str, t_m1_excitatory, t_m1_inhibitory]).cuda()

    def forward(self, inp, hn, x):

        '''
            Forward pass through the model
            
            Parameters:
                inp: input sequence, should be scalar values denoting the target time
                hn: the hidden state of the model
                x: hidden state before activation
        '''

        # Saving hidden states
        hn_next = hn.squeeze(0)
        x_next = x.squeeze(0)
        size = inp.shape[1]
        new_hs = []
        new_xs = []

        # Get full weights for training
        str2str_rec = (self.str2str_mask * F.relu(self.str2str_weight_l0_hh) + self.str2str_fixed) @ self.str2str_D
        m12m1_rec = F.relu(self.m12m1_weight_l0_hh) @ self.m12m1_D
        m12str_rec = self.m12str_mask * F.relu(self.m12str_weight_l0_hh)
        str2m1_rec = F.relu(self.str2m1_weight_l0_hh)

        # Concatenate into single weight matrix
        W_str = torch.cat([str2str_rec, m12str_rec], dim=1)
        W_alm = torch.cat([str2m1_rec, m12m1_rec], dim=1)
        W_rec = torch.cat([W_str, W_alm], dim=0)

        # Loop through RNN
        for t in range(size):
            x_next = (1 - self.t_const) * x_next + self.t_const * ((W_rec @ hn_next.T).T + (inp[:, t, :] @ self.inp_weight * self.str_mask))
            hn_next = F.relu(x_next)
            new_hs.append(hn_next)
            new_xs.append(x_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        x_out = torch.stack(new_xs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)
        x_last = x_out[:, -1, :].unsqueeze(0)

        # Behavioral output layer
        out = torch.sigmoid(self.fc1(rnn_out * self.alm_mask))
        
        return out, hn_last, rnn_out, x_last, x_out
    

########## SINGLE RNN MODEL ##########

class RNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, out_dim):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.rnn = nn.RNN(hid_dim, hid_dim, batch_first=True)
        self.fc2 = nn.Linear(hid_dim, out_dim)
    
    def forward(self, x, hn):
        
        out = F.relu(self.fc1(x))
        out, _ = self.rnn(out, hn)
        out = self.fc2(out)

        return out
        
