import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

######### MULTIREGIONAL MODEL ##########

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain = 1)
        torch.nn.init.constant_(m.bias, 0)


class RNN_MultiRegional_SAC(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, action_scale, action_bias, device):
        super(RNN_MultiRegional_SAC, self).__init__()
        
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
        self.alm_mask = torch.cat([torch.zeros(size=(hid_dim,)), torch.zeros(size=(hid_dim,)), torch.ones(size=(hid_dim,))]).to(device)
        self.str_mask = torch.cat([torch.ones(size=(hid_dim,)), torch.zeros(size=(hid_dim,)), torch.zeros(size=(hid_dim,))]).to(device)
        self.zeros = torch.zeros(size=(hid_dim, hid_dim)).to(device)
        
        # Identity Matrix of 0.5 Not Trained
        self.str2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.str2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Mix of Excitatory and Inhibitory Connections
        self.m12m1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.m12str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2m1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        nn.init.uniform_(self.str2str_weight_l0_hh, 0, 0.001)
        nn.init.uniform_(self.str2thal_weight_l0_hh, 0, 0.001)
        nn.init.uniform_(self.m12m1_weight_l0_hh, 0, 0.001)
        nn.init.uniform_(self.m12m1_weight_l0_hh, 0, 0.001)
        nn.init.uniform_(self.thal2m1_weight_l0_hh, 0, 0.001)

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
        
        # STR to Thal masks
        self.str2thal_D_mask = torch.cat([torch.ones(size=(int(hid_dim/2),)), -1 * torch.ones(size=(int(hid_dim/2),))])
        self.str2thal_D = (torch.eye(hid_dim) * self.str2thal_D_mask).to(device)
        
        # Input weights
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim * 3)))
        nn.init.uniform_(self.inp_weight, -0.1, 0.1)

        # Behavioral output layer
        self.mean_linear = nn.Linear(hid_dim * 3, action_dim)
        self.std_linear = nn.Linear(hid_dim * 3, action_dim)

        # Time constants for networks (not sure what would be biologically plausible?)
        self.t_const = 0.1
            

    

    def forward(self, inp, hn, sampling, len_seq = None):
       
        '''
            Forward pass through the model
            
            Parameters:
                inp: input sequence, should be scalar values denoting the target time
                hn: the hidden state of the model
                x: hidden state before activation
        '''
        
        # Saving hidden states
       
        hn_next = hn.squeeze(0)
        
        if sampling == True:
            size = inp.shape[1]
        
        
        new_hs = []
        #for batching
        rnn_out = []
        batch_hn_out = []

        # Get full weights for training
        str2str_rec = (self.str2str_mask * F.hardtanh(self.str2str_weight_l0_hh, min_val=1e-10, max_val = 1) + self.str2str_fixed) @ self.str2str_D
        m12m1_rec = F.hardtanh(self.m12m1_weight_l0_hh, min_val=1e-10, max_val = 1) @ self.m12m1_D
        m12str_rec = self.m12str_mask * F.hardtanh(self.m12str_weight_l0_hh, min_val=1e-10, max_val = 1)
        str2thal_rec = F.hardtanh(self.str2thal_weight_l0_hh, min_val=1e-10, max_val = 1) @ self.str2thal_D ##
        thal2m1_rec = F.hardtanh(self.thal2m1_weight_l0_hh, min_val=1e-10, max_val = 1)

        # Concatenate into single weight matrix

                            # STR         Thal        Cortex
        W_str = torch.cat([str2str_rec, self.zeros, m12str_rec], dim=1)     # STR
        W_thal = torch.cat([str2thal_rec, self.zeros, self.zeros], dim=1)          # Thal
        W_m1 = torch.cat([self.zeros, thal2m1_rec, m12m1_rec], dim=1)       # Cortex
        W_rec = torch.cat([W_str, W_thal, W_m1], dim=0)
 

        
        if sampling == False:
            #assert len_seq != None, "Proved the len_seq"
            #inp = pack_padded_sequence(inp, len_seq, batch_first = True, enforce_sorted = False)

            #pass from agent training, I just haven't done it yet
            batch_size = 8
            for i in range(batch_size): 
                state_batch = inp[i]
                for j in range(len_seq[i]):
                    hn_next = F.relu((1 - self.t_const) * hn_next + self.t_const*((W_rec @ hn_next.T).T + (state_batch[i, :]@self.inp_weight)))
                    new_hs.append(hn_next)
                rnn_out0 = torch.stack(new_hs, dim = 1).squeeze()
                rnn_out.append(rnn_out0)
                hn_last = rnn_out0[-1, :]
                batch_hn_out.append(hn_last)
                new_hs = []

            #pad batch outputs
            rnn_out = torch.FloatTensor(pad_sequence(rnn_out, batch_first = True)).to(self.device)
            
                                 
                
            

        if sampling == True:
            for t in range(size):
                hn_next = F.relu((1 - self.t_const) * hn_next + self.t_const * ((W_rec @ hn_next.T).T + (inp[:,t,:] @ self.inp_weight)))
                new_hs.append(hn_next)     
                rnn_out = torch.stack(new_hs, dim = 1)  
                hn_last = rnn_out[:, -1, :].unsqueeze(0)
       
     
        # Behavioral output layer
        mean_out = self.mean_linear(rnn_out * self.alm_mask)
        std_out = self.std_linear(rnn_out * self.alm_mask)
        std_out = torch.clamp(std_out, min = -5, max = 10)

    
        return mean_out, std_out, hn_last, rnn_out
    
    def sample(self, state, h_activity, sampling, len_seq):
        
        epsilon = 1e-4    
        
        mean, log_std, h_current, x = self.forward(state, h_activity, sampling, len_seq) #done
        

        if sampling == False:
            
            assert mean.size()[1] == log_std.size()[1], "There is a mismatch between mean and sigma S1_max"
            
            sl_max = mean.size()[1] 
            with torch.no_grad():
                for seq_idx, k in enumerate(len_seq):
                    for j in range(1, sl_max + 1):
                        if j <= k:
                            if seq_idx == 0 and j == 1:
                                mask_seq = torch.tensor([True], dtype = bool)
                            else:
                                mask_seq = torch.cat((mask_seq, torch.tensor([True])), dim = 0)
                        else:
                            mask_seq = torch.cat((mask_seq, torch.tensor([False])), dim = 0)

            mean = mean.reshape(-1, mean.size()[-1])[mask_seq]
            log_std = log_std.reshape(-1, log_std.size()[-1])[mask_seq]                   

        if sampling == True:

            mask_seq = [] 
        

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

        
        

        return action, log_prob, mean, h_current, mask_seq, x

    

########## SINGLE RNN MODEL ##########

class RNN_SAC(nn.Module):
    def __init__(self, inp_dim, action_dim, hid_dim):
        super(RNN_SAC, self).__init__()


        #Q1
        self.crt11 = nn.Linear(inp_dim + action_dim, hid_dim)
        self.crt12 = nn.Linear(hid_dim, hid_dim)
        self.crt13 = nn.Linear(hid_dim, 1)

        #Q2
        self.crt21 = nn.Linear(inp_dim + action_dim, hid_dim)
        self.crt22 = nn.Linear(hid_dim, hid_dim)
        self.crt23 = nn.Linear(hid_dim, 1)

        self.apply(weights_init_)
    
    def forward(self, state, action):
       
        

        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.crt11(xu))
        x1 = F.relu(self.crt12(x1))
        x1 = self.crt13(x1)

        x2 = F.relu(self.crt21(xu))
        x2 = F.relu(self.crt22(x2))
        x2 = self.crt23(x2)
        
       

        return x1, x2
        
