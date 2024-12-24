import os
import sys
import json
import numpy as np
import torch 
import matplotlib.pyplot as plt
import motornet as mn
from motornet_env import EffectorTwoLinkArmEnv
from models import RNN

class Optimizer_Policy(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.activity = np.array([])
        self.count = 0
        
        self.gru = torch.nn.GRU(input_dim, hidden_dim, 1, batch_first = False)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                torch.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                torch.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                torch.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                torch.nn.init.zeros_(param)
            elif name == "fc.weight":
                torch.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                torch.nn.init.constant_(param, -5.)
            else:
                raise ValueError
        
        self.to(device)

    def forward(self, x, h0):
        y, h = self.gru(x, h0)
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        '''
        u_ = u.clone().norm()
        self.activity = np.append(self.activity, u_.detach().numpy())
        self.count += 1
        if self.count == 5000:
            print(self.activity)
            plt.plot(self.activity)
            plt.show()
            '''
        return u, h
    
    


class optimizer():
    def __init__(self,
                 env,
                 batch_size,
                 inp_dim,
                 hid_dim,
                 action_dim,
                 action_scale,
                 action_bias):
        
        self.env = env
        self.batch_size = batch_size
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"



        self.actor = Optimizer_Policy(self.inp_dim, self.hid_dim, self.action_dim, device=self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=10**-4)
        self.losses = []
        self.interval = 250

    def get_loss(self, xy, tg):
        #calculate loss as a function of distance from target
        loss = torch.mean(torch.sum(torch.abs(xy - tg), dim = -1))
        return loss

    

    def train(self, max_episodes, test_train):
        episode = 0
        all_rewards = []
        interval = 250
        
        
        for t in range(max_episodes):

            done = False
            episode_steps = 0
            

            #reset environment 
            #returns state of : target [x,y] , joint angles and velocity [a,b,c,d], fingertip positions [d,f]
            h = torch.zeros(size = (1,self.hid_dim), device = self.device )

            state = self.env.reset(episode)

            #get x,y position (fingertip)
            xy = [state[:, -2:]]

            #get target position 
            tg = [state[:, 0:2]]

            while not done:
                
                #get action from policy
                action, h = self.actor(state, h)

                #Take a step to get next state and done
                episode_steps += 1
                state, reward, done = self.env.step(episode_steps, action, episode)
            

                #append position
                xy.append(state[:, -2:])
                #append target
                tg.append(state[:, 0:2])
            
            
            #convert positions and target to tensor
            xy = torch.cat(xy, dim = 0)
            tg = torch.cat(tg, dim = 0)

            all_rewards.append(reward)
            episode += 1

            #get loss
            loss = self.get_loss(xy,tg)
        

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm = 1.)
            self.optimizer.step()
            self.losses.append(loss.item())

        
            if done:
                if episode % 25 == 0:
                    print(f'Batch {episode} Done, average reward: {sum(all_rewards)/episode}. mean loss : {sum(self.losses)/episode}')


'''
p
 Questions
    sigmoid not relu?


    Remember to:
    add argument for max number of episodes, maybe change SAC to that as well instead of max timesteps


    initialize fully connected layer bias to be -5
'''

       