import os
import sys
import json
import numpy as np
import torch 
import matplotlib.pyplot as plt
import motornet as mn
from motornet_env import EffectorTwoLinkArmEnv
from models import RNN

class optimizer():
    def __init__(self,
                 env,
                 batch_size,
                 inp_dim,
                 hid_dim,
                 action_dim,
                 action_scale,
                 action_bias,
                 optimizer_spec_actor):
        
        self.env = env
        self.batch_size = batch_size
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer_spec_actor = optimizer_spec_actor

        self.actor = RNN(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias, self.device).to(self.device)
        self.actor_optimizer = self.optimizer_spec_actor.constructor(self.actor.parameters(), **self.optimizer_spec_actor.kwargs) 
        self.losses = []
        self.interval = 250

    def get_loss(self, xy, tg):
        #calculate loss as a function of distance from target
        loss = torch.mean(torch.sum(torch.abs(xy - tg), dim = -1))
        return loss


    def select_action(self, state, h_prev, iteration , iteration0, evaluate):
        
        state = torch.tensor(state, dtype = torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        h_prev = h_prev.to(self.device)

        #For training
        if evaluate == False:
            action, _, mean, rnn_out, h_current, std = self.actor.sample(state, h_prev, iteration, iteration0)
            mean = mean.squeeze().cpu().numpy()
            std = std.squeeze().cpu().numpy()
            return action.squeeze().detach().cpu().numpy(), h_current.detach(), rnn_out.detach().cpu().numpy(), mean, std 
        
        #For testing
        if evaluate == True:
            _, _, action, rnn_out, h_current, std= self.actor.sample(state, h_prev, iteration, iteration0)
            return action, h_current
    

    def train(self, max_episodes, test_train):
        episode = 0
        all_rewards = []
        
        for t in range(max_episodes):

            done = False
            episode_steps = 0
            

            #reset environment 
            #returns state of : target [x,y] , joint angles and velocity [a,b,c,d], muscle activations [1,2,3,4,5,6] fingertip positions [d,f]
            h_prev = torch.zeros(size = (1 ,1 , self.hid_dim), device = self.device )
            state = self.env.reset(episode)

            #get x,y position (fingertip)
            xy = torch.tensor(state[-2:], dtype=torch.float32, requires_grad = True)
            #get target position 
            tg = torch.tensor(state[0:2], dtype=torch.float32, requires_grad = True)

            while not done:
                
                #get action from policy
                with torch.no_grad():
                    action, h_current, _, _, _ = self.select_action(state, h_prev, iteration = episode, iteration0 = None, evaluate = False)
                
                #Take a step to get next state and done
                episode_steps += 1
                next_state, reward, done = self.env.step(episode_steps, action, h_current, episode)

                #append position
                torch.cat((xy, torch.tensor(next_state[-2:], requires_grad = True )))

                #append target
                torch.cat((tg, torch.tensor(next_state[0:2], requires_grad = True))) 

            #convert positions and target to tensors
            all_rewards.append(reward)
            episode += 1

            #get loss
            loss = self.get_loss(xy,tg)

            self.actor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm = 1.)
            self.actor_optimizer.step()
            self.losses.append(loss.item())

            if done:
                print(f'Batch {episode} Done, average reward: {sum(all_rewards)/episode}. mean loss : {sum(self.losses)/episode}')


'''
 Questions
    sigmoid not relu?

    init hidden vs our initialization?

    initialize at every episode?

    Remember to:
    add argument for max number of episodes, maybe change SAC to that as well instead of max timesteps


    initialize fully connected layer bias to be -5
'''

       