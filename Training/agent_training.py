import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import os 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from reward_vis import *
from policy_replay import PolicyReplayBuffer
from torch.optim import Adam

from models import RNN_MultiRegional, RNN, Critic, Critic2
import scipy.io as sio
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
    
class Off_Policy_Agent():
    def __init__(self, 
                 policy_replay_size,
                 policy_batch_size, 
                 policy_batch_iters,
                 lr,
                 alpha,
                 env,
                 seed,
                 inp_dim,
                 hid_dim,
                 action_dim,
                 optimizer_spec_actor,
                 optimizer_spec_critic,
                 tau,
                 gamma,
                 save_iter,
                 log_steps,
                 frame_skips,
                 model_save_path,
                 buffer_save_path,
                 reward_save_path,
                 vis_save_path,
                 action_scale,
                 action_bias,
                 automatic_entropy_tuning,
                 continue_training,
                 test_train):
        

        self.policy_replay_size = policy_replay_size
        self.policy_batch_size = policy_batch_size
        self.policy_batch_iters = policy_batch_iters
        self.alpha = alpha
        self.alpha_vis = np.array([alpha])
        self.env = env
        self.seed = seed
        self.inp_dim = inp_dim
        self.hid_dim =hid_dim
        self.action_dim = action_dim     
        self.optimizer_spec_actor = optimizer_spec_actor
        self.optimizer_spec_critic = optimizer_spec_critic
        self.tau = tau
        self.gamma = gamma
        self.save_iter = save_iter
        self.log_steps = log_steps
        self.frame_skips = frame_skips
        self.model_save_path = model_save_path
        self.buffer_save_path = buffer_save_path
        self.reward_save_path = reward_save_path
        self.vis_save_path = vis_save_path
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.automatic_entropy_tuning = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_train = test_train

        self.policy_memory = PolicyReplayBuffer(self.policy_replay_size, self.seed)

        self.lr = lr

        #initialize Actor/Critic RNNs 
        
        self.actor = RNN(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias, self.device).to(self.device)

        self.critic = Critic2(self.inp_dim, self.action_dim, self.hid_dim).to(self.device)

        self.target_critic = Critic2(self.inp_dim, self.action_dim, self.hid_dim).to(self.device)

        #Optimizers now defined for each Q network seperately
        self.actor_optimizer = self.optimizer_spec_actor.constructor(self.actor.parameters(), **self.optimizer_spec_actor.kwargs) 

        self.critic_optimizer = self.optimizer_spec_critic.constructor(self.critic.parameters(), **self.optimizer_spec_critic.kwargs)

        if continue_training == "yes":
            
            checkpoint = torch.load(f'{self.model_save_path}.pth', map_location = torch.device('cpu'))

            #load in models
            self.actor.load_state_dict(checkpoint['agent_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

            #load in optimizers 
            self.actor_optimizer.load_state_dict(checkpoint['agent_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            #load in buffer
            self.policy_memory.buffer = checkpoint['buffer']


        #update critics and their targets
        if continue_training != "yes":
            self.hard_update(self.target_critic, self.critic)


        #target entropy = - dim(action)
        if self.automatic_entropy_tuning:
            print("using automatic entropy tuning")
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad = True, device = self.device)
            self.alpha_optim = Adam([self.log_alpha], lr = self.lr)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def select_action(self, state, h_prev, evaluate):
        
        state = torch.tensor(state, dtype = torch.float32, device=self.device).unsqueeze(0)
        h_prev = h_prev.to(self.device)

        #For training
        if evaluate == False:
            action, _, mean, rnn_out, h_current, std = self.actor.sample(state, h_prev)
            mean = mean.squeeze().cpu().numpy()
            std = std.squeeze().cpu().numpy()
            return action.squeeze().detach().cpu().numpy(), h_current.detach(), rnn_out.detach().cpu().numpy(), mean, std 
        
        #For testing
        if evaluate == True:
            _, _, action, rnn_out, h_current, std= self.actor.sample(state, h_prev)
            return action, h_current

        


    def test(self, max_steps):
        print('testing')
        
        checkpoint = torch.load(f'{self.model_save_path}.pth', map_location = torch.device('cpu'))
        self.actor = RNN(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias, self.device).to(self.device) #change to self actor
        self.actor.load_state_dict(checkpoint['agent_state_dict'])

        iteration = checkpoint['iteration']
        iteration0 = iteration

        #Initializing...
        state = self.env.reset(episode)
        h_prev = torch.zeros(size=(1 ,1 , self.hid_dim), device = self.device)
        num_episodes = 0
        episode_steps = 0 
        episode_reward = 0 
        count = 0 
        test = True

        #run episode as usual in train but without the update
        for t in range(self.env.max_timesteps):

            with torch.no_grad():
                action, h_current = self.select_action( state, h_prev, evaluate = True)
            
            for _ in range(self.frame_skips):

                episode_steps += 1
                next_state, reward, done = self.env.step(episode_steps, action, h_current)
                episode_reward += reward[0]
                if done:
                    break
                
            state = next_state
            h_prev = h_current 

            if done:
                print("episode steps", episode_steps)

                state = self.env.reset(iteration)

                h_prev = torch.zeros(size = (1 , 1, self.hid_dim), device = self.device)

                episode_reward = 0
                episode_steps = 0 

                break

        if vis_activity:
            activity_vis(self.reward_save_path, self.vis_save_path, True)


    def train(self, max_steps):

        #Initialize tracking statistics 
        Statistics = {'best_mean_episode_rewards' : [],
                     'mean_episode_rewards' : [],
                     'mean_episode_steps' : [],
                     'all_episode_steps' : [],
                     'all_episode_rewards' : [],
                     'actor_gradients' : [],
                     'critic_gradients' : [],
                     'critic_loss': [],
                     'critic_target_loss': [],
                     'actor_loss': [],
                     'sampled_entropies': [],
                     'batch_entropies': [],
                     'mean': [],
                     'std': [], 
                     'alpha': []
                     }

        total_episodes = 0
        best_mean_episode_reward = -float("inf")
        all_steps = []
        all_reward = []
        actor_losses = []
        critic_losses = []
        critic_target_losses = []
        sampled_entropies = []
        batch_entropies = []
        ep_trajectory = []
        state = self.env.reset(0)

        episode_steps = 0
        episode_reward = 0

        grad_vis_actor = {}
        grad_vis_critic = {}

        for name, param in self.actor.named_parameters():
            grad_vis_actor[f'{name}'] = []

        for name, param in self.critic.named_parameters():
            grad_vis_critic[f'{name}'] = []

        h_prev = torch.zeros(size = (1 ,1 , self.hid_dim), device = self.device )

        #Episode Training Loop
        for t in range(max_steps):

            #add to replay buffer 
            if len(self.policy_memory.buffer) > self.policy_batch_size:
                for _ in range(self.policy_batch_iters):
                    self.update() 

            with torch.no_grad():   
                action, h_current, _, mean, std = self.select_action(state, h_prev, evaluate = False)
          

            for _ in range(self.frame_skips):
                episode_steps += 1
                next_state, reward, done = self.env.step(episode_steps, action, total_episodes)
                episode_reward += reward
                if done == True:
                    break

            mask = 1.0 if episode_steps == self.env.max_timesteps else float(not done)

            ep_trajectory.append((state, action, reward, next_state, mask))

            state = next_state
            h_prev = h_current         

            Statistics["mean"].append(mean)
            Statistics["std"].append(std)
            Statistics["alpha"].append(self.alpha_vis)

            if done:

                total_episodes += 1
                all_steps.append(episode_steps)
                all_reward.append(episode_reward)

                #push to replay and update
                self.policy_memory.push(ep_trajectory)

                ### 4. Log progress and keep track of statistics
                if len(all_reward) > 0:
                    mean_episode_reward = np.mean(np.array(all_reward)[-1000:])
                if len(all_steps) > 0:
                    mean_episode_steps = np.mean(np.array(all_steps)[-1000:])
                if len(all_reward) > 10:
                    if mean_episode_reward >= best_mean_episode_reward: ##Save best reward instead
                        torch.save({
                            'iteration': t,
                            'agent_state_dict': self.actor.state_dict(),
                            'critic_state_dict': self.critic.state_dict(),
                            'target_critic_state_dict': self.target_critic.state_dict(),
                            'agent_optimizer_state_dict': self.actor_optimizer.state_dict(),
                            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                            'buffer' : self.policy_memory.buffer
                        }, self.model_save_path + '.pth')

                    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                #update gradient log
                Statistics["mean_episode_rewards"].append(mean_episode_reward)
                Statistics["mean_episode_steps"].append(mean_episode_steps)
                Statistics["best_mean_episode_rewards"].append(best_mean_episode_reward)
                Statistics["all_episode_rewards"] = all_reward
                Statistics["all_episode_steps"] = all_steps
                Statistics["actor_gradients"] = grad_vis_actor
                Statistics["critic_gradients"] = grad_vis_critic
                Statistics["actor_loss"] = actor_losses
                Statistics["critic_loss"] = critic_losses
                Statistics["critic_target_loss"] = critic_target_losses
                Statistics["sampled_entropies"] = sampled_entropies
                Statistics["batch_entropies"] = batch_entropies

                print("Episode %d" % (total_episodes,))
                print("reward: %f" % episode_reward)
                print("steps: %f" % episode_steps)
                print("best mean reward: %f" % best_mean_episode_reward)
                sys.stdout.flush()

                if total_episodes % self.log_steps == 0:
                    # Dump statistics to pickle
                    np.save(f'{self.reward_save_path}.npy', Statistics)
                    print("Saved to %s" % 'training_reports')

                if total_episodes % 1000 == 0: #save graphs every n episodes
                    average_reward_vis(f'{self.reward_save_path}.npy', self.vis_save_path)

                #Reset lists and activity
                episode_reward = 0
                episode_steps = 0
                
                ep_trajectory = []

                h_prev = torch.zeros(size = (1 ,1, self.hid_dim), device = self.device)
                state = self.env.reset(total_episodes)

        
    def update(self):

        #Sample from replay memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.policy_memory.sample(self.policy_batch_size)

        # Get output mask for loss
        len_seq = list(map(len, state_batch))
        mask = [torch.ones(size=(length,)) for length in len_seq]
        mask = pad_sequence(mask, batch_first=True).unsqueeze(-1).to(self.device)

        #Pad sequences for forward pass
        state_batch = pad_sequence(state_batch, batch_first=True).to(self.device)
        action_batch = pad_sequence(action_batch, batch_first=True).to(self.device)
        reward_batch = pad_sequence(reward_batch, batch_first=True).to(self.device)
        next_state_batch = pad_sequence(next_state_batch, batch_first=True).to(self.device)
        mask_batch = pad_sequence(mask_batch, batch_first=True).unsqueeze(-1).to(self.device)
      

        #Activites for sampling
        h0_actor = torch.zeros(size=(1, next_state_batch.shape[0], self.hid_dim)).to(self.device)
        h0_critic = torch.zeros(size=(1, next_state_batch.shape[0], self.hid_dim)).to(self.device)


        ##Critic Loss##

        #Calculate target q using action sampled from policy and next state from batch
        with torch.no_grad():
            next_action, next_log_prob, _, _, _, _, _ = self.actor.sample(next_state_batch, h0_actor, iteration = None, iteration0 = None)
            qf1_next_target, qf2_next_target = self.target_critic(next_state_batch, next_action, h0_critic)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            target_q = reward_batch + mask_batch * self.gamma*(min_qf_next_target - self.alpha * next_log_prob) 
            target_q = mask * target_q


        #Calculate q using batch state and batch action
        qf1, qf2 = self.critic(state_batch, action_batch, h0_critic)

        #Mask critic outputs
        qf1 = mask * qf1  
        qf2 = mask * qf2
    
        #Calculate Critic loss (MSE(q and target))
        qf1_loss = F.mse_loss(qf1, target_q)
        qf2_loss = F.mse_loss(qf2, target_q)
        qf_loss = qf1_loss + qf2_loss
        
        #Take Gradient Steps for Q functions
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        for name, param in self.critic.named_parameters():
            norm_grad = np.array(torch.norm(param).detach().cpu())
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        ##Policy Update##
        #Sample reparameterized actions from state batch
        
        reparam_action, log_prob_batch, _, _, _, _, _ = self.actor.sample(state_batch, h0_actor, iteration = None, iteration0 = None)
        reparam_action = mask * reparam_action
        log_prob_batch = mask * log_prob_batch

        #Get Q Value of Current State and Action Pairs
        qf1_pi, qf2_pi = self.critic(state_batch, reparam_action, h0_critic)
        qf1_pi = mask * qf1_pi
        qf2_pi = mask * qf2_pi

        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        #Policy loss
        policy_loss = ((self.alpha * log_prob_batch) - min_qf_pi).mean() #+ torch.mean(hn_next, dim = -1)*.01

        #Policy Gradient Step
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        for name, param in self.actor.named_parameters():
            norm_grad = np.array(torch.norm(param).detach().cpu())

        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        #Automatic Entropy Tuning
        log_pi = log_prob_batch
        if self.automatic_entropy_tuning:

            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        #Soft Update Actor Critic
        self.soft_update(self.target_critic, self.critic, self.tau)

        

        