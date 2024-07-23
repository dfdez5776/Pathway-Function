import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

######### MULTIREGIONAL MODEL ##########

class RNN_MultiRegional(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, action_scale, action_bias, device):
        super(RNN_MultiRegional, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                inp_dim: dimension of input
                hid_dim: number of hidden neurons, each region and connection between region has hid_dim/2 neurons
                action_dim: output dimension, should be one for lick or no lick
        '''

        # Network Variables
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.action_scale = action_scale 
        self.action_bias = action_bias
        self.device = device

        # Masks for individual regions
        self.m1_mask = torch.cat([torch.zeros(size=(hid_dim*4,)), torch.zeros(size=(hid_dim,))]).to(device)
        self.thal_mask = torch.cat([torch.zeros(size=(hid_dim*3,)), torch.ones(size=(hid_dim,)), torch.zeros(size=(hid_dim,))]).to(device)
        self.str_mask = torch.cat([torch.ones(size=(hid_dim,)), torch.zeros(size=(hid_dim*4,))]).to(device)
        self.zeros = torch.zeros(size=(hid_dim, hid_dim)).to(device)

        # Inhibitory Connections
        self.str2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.str2snr_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Mix of Excitatory and Inhibitory Connections
        self.m12m1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.m12str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2m1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.m12thal_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.str2stn_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.stn2snr_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.snr2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        nn.init.uniform_(self.str2str_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.str2snr_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.m12m1_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.m12str_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.thal2m1_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.m12thal_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.thal2str_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.str2stn_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.stn2snr_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.snr2thal_weight_l0_hh, 0, 0.01)

        # Implement Necessary Masks
        # Striatum recurrent weights
        sparse_matrix = torch.empty_like(self.str2str_weight_l0_hh)
        nn.init.sparse_(sparse_matrix, 0.85)
        sparse_mask = torch.where(sparse_matrix != 0, 1, 0)
        self.str2str_mask = torch.zeros_like(self.str2str_weight_l0_hh).to(device)
        self.str2str_fixed = (torch.empty_like(self.str2str_weight_l0_hh).uniform_(0, 0.01) * sparse_mask).to(device)
        self.str2str_D = -1*torch.eye(hid_dim).to(device)

        self.m12m1_D = torch.eye(hid_dim).to(device)
        self.m12m1_D[hid_dim-(int(0.3*hid_dim)):, 
                        hid_dim-(int(0.3*hid_dim)):] *= -1
        
        # ALM to striatum weights
        self.m12str_mask_excitatory = torch.ones(size=(hid_dim, hid_dim - int(0.3*hid_dim))).to(device)
        self.m12str_mask_inhibitory = torch.zeros(size=(hid_dim, int(0.3*hid_dim))).to(device)
        self.m12str_mask = torch.cat([self.m12str_mask_excitatory, self.m12str_mask_inhibitory], dim=1).to(device)

        # M1 to thal mask
        self.m12thal_mask = torch.cat([torch.zeros(size=(int(hid_dim/2), hid_dim)),
                                       torch.ones(size=(int(hid_dim/2), hid_dim))]).to(device)

        # STR to STN mask
        self.str2stn_mask = torch.cat([torch.zeros(hid_dim, int(hid_dim/2)),
                                       torch.ones(hid_dim, int(hid_dim/2))], dim=1).to(device)
        
        # STN to SNr D
        self.stn2snr_D = -1 * torch.eye(hid_dim).to(device)

        # SNr to Thal D
        self.snr2thal_D = -1 * torch.eye(hid_dim).to(device)
        
        # STR to SNr D and mask
        self.str2snr_D = -1 * torch.eye(hid_dim).to(device)
        self.str2snr_mask = torch.cat([torch.ones(hid_dim, int(hid_dim/2)),
                                       torch.zeros(hid_dim, int(hid_dim/2))], dim=1).to(device)
        
        # Input weights
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim * 5)))
        nn.init.uniform_(self.inp_weight, 0, 0.1)

        # Behavioral output layer
        self.mean_linear = nn.Linear(hid_dim * 5, action_dim)
        self.std_linear = nn.Linear(hid_dim * 5, action_dim)

        # Time constants for networks
        self.t_const = 0.1

    def forward(self, inp, hn):

        '''
            Forward pass through the model
            
            Parameters:
                inp: input sequence, should be scalar values denoting the target time
                hn: the hidden state of the model
                x: hidden state before activation
        '''

        # Saving hidden states
        hn_next = hn.squeeze(0)      
        size = inp.shape[1]
        new_hs = []

        # Get full weights for training
        str2str_rec = (self.str2str_mask * F.hardtanh(self.str2str_weight_l0_hh, min_val=1e-10, max_val=1) + self.str2str_fixed) @ self.str2str_D
        m12m1_rec = F.hardtanh(self.m12m1_weight_l0_hh, min_val=1e-10, max_val=1) @ self.m12m1_D
        m12str_rec = self.m12str_mask * F.hardtanh(self.m12str_weight_l0_hh, min_val=1e-10, max_val=1)
        str2snr_rec = self.str2snr_mask * F.hardtanh(self.str2snr_weight_l0_hh, min_val=1e-10, max_val=1) @ self.str2snr_D
        thal2m1_rec = F.hardtanh(self.thal2m1_weight_l0_hh, min_val=1e-10, max_val=1)
        m12thal_rec = F.hardtanh(self.m12thal_weight_l0_hh, min_val=1e-10, max_val=1)
        thal2str_rec = F.hardtanh(self.thal2str_weight_l0_hh, min_val=1e-10, max_val=1)
        str2stn_rec = self.str2stn_mask * F.hardtanh(self.str2stn_weight_l0_hh, min_val=1e-10, max_val=1)
        stn2snr_rec = F.hardtanh(self.str2stn_weight_l0_hh, min_val=1e-10, max_val=1) @ self.stn2snr_D
        snr2thal_rec = F.hardtanh(self.snr2thal_weight_l0_hh, min_val=1e-10, max_val=1) @ self.snr2thal_D
        inp_weight = F.hardtanh(self.inp_weight, 1e-10, 1)

        # Concatenate into single weight matrix

                            # STR           STN         SNr        Thal         Cortex
        W_str = torch.cat([str2str_rec, self.zeros, self.zeros, thal2str_rec, m12str_rec], dim=1)               # STR
        W_stn = torch.cat([str2stn_rec, self.zeros, self.zeros, self.zeros, self.zeros], dim=1)                 # STN
        W_snr = torch.cat([str2snr_rec, stn2snr_rec, self.zeros, self.zeros, self.zeros], dim=1)                # SNr
        W_thal = torch.cat([self.zeros, self.zeros, snr2thal_rec, self.zeros, m12thal_rec], dim=1)              # Thal
        W_m1 = torch.cat([self.zeros, self.zeros, self.zeros, thal2m1_rec, m12m1_rec], dim=1)                   # Cortex
        W_rec = torch.cat([W_str, W_stn, W_snr, W_thal, W_m1], dim=0)

        # Loop through RNN
        for t in range(size):
            hn_next = F.relu((1 - self.t_const) * hn_next + self.t_const * ((W_rec @ hn_next.T).T + (inp[:, t, :] @ inp_weight * self.thal_mask)))
            new_hs.append(hn_next)

        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)

        # Behavioral output layer
        mean_out = self.mean_linear(rnn_out * self.m1_mask)
        std_out = self.std_linear(rnn_out * self.m1_mask)
        std_out = torch.clamp(std_out, min = -5, max = 10)
    
        return mean_out, std_out, rnn_out, hn_last
    
    def sample(self, state, hn, sampling, len_seq):

        epsilon = 1e-4    
        
        mean, log_std, rnn_out, hn = self.forward(state, hn)

        mean_size = mean.size()
        log_std_size = log_std.size()

        mean = mean.reshape(-1, mean.size()[-1])
        log_std = log_std.reshape(-1, log_std.size()[-1])

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if sampling == False:
            action = action.reshape(mean_size[0], mean_size[1], mean_size[2])
            log_prob = log_prob.reshape(log_std_size[0], log_std_size[1], 1) 
            mean = mean.reshape(mean_size[0], mean_size[1], mean_size[2])

        return action, log_prob, mean, rnn_out, hn

########## SINGLE RNN MODEL ##########

class RNN(nn.Module):
    def __init__(self, inp_dim, action_dim, hid_dim):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.fc11 = nn.Linear(inp_dim + action_dim, hid_dim)
        self.rnn1 = nn.GRU(hid_dim, hid_dim, batch_first=True)
        self.fc12 = nn.Linear(hid_dim, 1)

        self.fc21 = nn.Linear(inp_dim + action_dim, hid_dim)
        self.rnn2 = nn.GRU(hid_dim, hid_dim, batch_first=True)
        self.fc22 = nn.Linear(hid_dim, 1)
    
    def forward(self, state, action, hn):

        xu = torch.cat([state, action], dim=-1)
        
        out1 = F.relu(self.fc11(xu))
        out1, _ = self.rnn1(out1, hn)
        out1 = self.fc12(out1)

        out2 = F.relu(self.fc21(xu))
        out2, _ = self.rnn2(out2, hn)
        out2 = self.fc22(out2)

        return out1, out2
        
