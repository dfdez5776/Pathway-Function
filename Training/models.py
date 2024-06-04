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

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.action_scale = action_scale 
        self.action_bias = action_bias
        self.device = device
        self.alm_mask = torch.cat([torch.zeros(size=(int(hid_dim/2),)), torch.ones(size=(int(hid_dim/2),))]).to(device)
        self.str_mask = torch.cat([torch.ones(size=(int(hid_dim/2),)), torch.zeros(size=(int(hid_dim/2),))]).to(device)
        
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
        sparse_mask = torch.where(sparse_matrix != 0, 1, 0)
        self.str2str_mask = torch.zeros_like(self.str2str_weight_l0_hh).to(device)
        self.str2str_fixed = (torch.empty_like(self.str2str_weight_l0_hh).uniform_(0, 0.01) * sparse_mask).to(device)
        self.str2str_D = -1*torch.eye(int(hid_dim/2)).to(device)

        self.m12m1_D = torch.eye(int(hid_dim/2)).to(device)
        self.m12m1_D[int(hid_dim/2)-(int( 0.3*(hid_dim/2) )):, 
                        int(hid_dim/2)-(int( 0.3*(hid_dim/2) )):] *= -1
        
        # ALM to striatum weights
        self.m12str_mask_excitatory = torch.ones(size=(int(hid_dim/2), int(hid_dim/2) - int(0.3*(hid_dim/2)))).to(device)
        self.m12str_mask_inhibitory = torch.zeros(size=(int(hid_dim/2), int(0.3*(hid_dim/2)))).to(device)
        self.m12str_mask = torch.cat([self.m12str_mask_excitatory, self.m12str_mask_inhibitory], dim=1).to(device)
        
        # Input weights
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim)))
        nn.init.uniform_(self.inp_weight, 0, 0.1)

        # Behavioral output layer
        self.mean_linear = nn.Linear(hid_dim, action_dim)
        self.std_linear = nn.Linear(hid_dim, action_dim)

        # Time constants for networks (not sure what would be biologically plausible?)
        t_str = 0.1 * torch.ones(int(hid_dim/2))
        t_m1_excitatory = 0.1 * torch.ones(int(hid_dim/2) - int(0.3*(hid_dim/2)))
        t_m1_inhibitory = 0.1 * torch.ones(int(0.3*(hid_dim/2)))
        self.t_const = torch.cat([t_str, t_m1_excitatory, t_m1_inhibitory]).to(device)

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
        mean_out = self.mean_linear(rnn_out * self.alm_mask)
        std_out = self.std_linear(rnn_out * self.alm_mask)

        return mean_out, std_out, rnn_out, hn_last, x_last, x_out
    
    def sample(self, state, hn, x, sampling):

        epsilon = 1e-4   

        mean, log_std, rnn_out, hn, x_last, x_out = self.forward(state, hn, x)

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

        return action, log_prob, mean, rnn_out, hn, x_last, x_out

    

########## SINGLE RNN MODEL ##########

class RNN(nn.Module):
    def __init__(self, inp_dim, hid_dim):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.rnn = nn.RNN(hid_dim, hid_dim, batch_first=True)
        self.fc2 = nn.Linear(hid_dim, 1)
    
    def forward(self, x, hn):
        
        out = F.relu(self.fc1(x))
        out, _ = self.rnn(out, hn)
        out = self.fc2(out)

        return out
        
