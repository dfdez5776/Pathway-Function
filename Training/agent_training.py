import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

#from utils.gym import get_wrapper_by_name  
from models import RNN_MultiRegional, RNN  #Changed Actor to RNN_Multiregional(nn.Module), Value to RNN, not sure for Critic
import scipy.io as sio
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class On_Policy_Agent():
    def __init__(self, 
                env,
                seed,
                inp_dim,
                hid_dim,
                action_dim,
                optimizer_spec_actor,
                optimizer_spec_critic,
                gamma,
                save_iter,
                log_steps,
                frame_skips,
                model_save_path,
                reward_save_path,
                steps_save_path,
                action_scale,
                action_bias):

        self.env = env
        self.seed = seed
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.optimizer_spec_actor = optimizer_spec_actor
        self.optimizer_spec_critic = optimizer_spec_critic
        self.gamma = gamma
        self.save_iter = save_iter
        self.log_steps = log_steps
        self.frame_skips = frame_skips
        self.model_save_path = model_save_path
        self.reward_save_path = reward_save_path
        self.steps_save_path = steps_save_path
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _Normalize_Data(self, data, min, max):
        '''
            Mainly used for neural data if model is constrained
        '''
        return (data - min) / (max - min)

    def select_action(self, policy, state, hn, x, evaluate):
        '''
            Selection of action from policy, consistent across training methods
        '''
        
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
        hn = hn.to(self.device)
        x = x.to(self.device)

        if evaluate == False: 
            action, _, _, _, hn, x, _ = policy.sample(state, hn, x, sampling=True)
        else:
            _, _, action, _, hn, x, _ = policy.sample(state, hn, x, sampling=True)

        return action.detach().cpu().numpy(), hn.detach(), x.detach()
    
    def train(self, max_steps):

        '''
            Train the agent using one step actor critic
        '''

        actor_bg = RNN_MultiRegional(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias, self.device).to(self.device)
        critic_bg = RNN(self.inp_dim, self.hid_dim).to(self.device)

        actor_bg_optimizer = self.optimizer_spec_actor.constructor(actor_bg.parameters(), **self.optimizer_spec_actor.kwargs)
        critic_bg_optimizer = self.optimizer_spec_critic.constructor(critic_bg.parameters(), **self.optimizer_spec_critic.kwargs)

        z_actor = {}
        z_critic = {}
        I = 1
        for name, params in actor_bg.named_parameters():
            z_actor[name] = torch.zeros_like(params)
        for name, params in critic_bg.named_parameters():
            z_critic[name] = torch.zeros_like(params)

        Statistics = {
            "mean_episode_rewards": [],
            "mean_episode_steps": [],
            "best_mean_episode_rewards": []
        }

        episode_reward = 0
        best_mean_episode_reward = -float("inf")
        episode_steps = 0
        total_episodes = 0
        avg_reward = [0]
        avg_steps = [0]

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset(0)
        ep_trajectory = []

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, self.hid_dim), device=self.device)
        
        x_prev = torch.zeros(size=(1, 1, self.hid_dim), device=self.device)

        ### STEPS PER EPISODE ###
        for t in range(max_steps):

            with torch.no_grad():
                action, h_current, x_current = self.select_action(actor_bg, state, h_prev, x_prev, evaluate=False)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(self.frame_skips):
                next_state, reward, done = self.env.step(episode_steps, action)
                episode_steps += 1
                episode_reward += reward[0]
                if done == True:
                    break

            mask = 1.0 if episode_steps == self.env.max_timesteps else float(not done)

            ep_trajectory.append((state, action, reward, next_state, mask))

            state = next_state
            h_prev = h_current
            x_prev = x_current 

            ### EARLY TERMINATION OF EPISODE
            if done:

                total_episodes += 1

                # Add stats to lists
                avg_steps.append(episode_steps)
                avg_reward.append(episode_reward)

                # reset training conditions
                h_prev = torch.zeros(size=(1, 1, self.hid_dim), device=self.device)
                x_prev = torch.zeros(size=(1, 1, self.hid_dim), device=self.device)
                state = self.env.reset(total_episodes) 

                # resest lists
                ep_trajectory = []

                # reset eligibility trace (if using on-policy method)
                z_actor = {}
                z_critic = {}
                I = 1
                for name, params in actor_bg.named_parameters():
                    z_actor[name] = torch.zeros_like(params)
                for name, params in critic_bg.named_parameters():
                    z_critic[name] = torch.zeros_like(params)

                ### 4. Log progress and keep track of statistics
                if len(avg_reward) > 0:
                    mean_episode_reward = np.mean(np.array(avg_reward)[-1000:])
                if len(avg_steps) > 0:
                    mean_episode_steps = np.mean(np.array(avg_steps)[-1000:])
                if len(avg_reward) > 10:
                    if mean_episode_reward > best_mean_episode_reward:
                        torch.save({
                            'iteration': t,
                            'agent_state_dict': actor_bg.state_dict(),
                            'critic_state_dict': critic_bg.state_dict(),
                            'agent_optimizer_state_dict': actor_bg_optimizer.state_dict(),
                            'critic_optimizer_state_dict': critic_bg_optimizer.state_dict(),
                        }, self.model_save_path + '.pth')

                    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                Statistics["mean_episode_rewards"].append(mean_episode_reward)
                Statistics["mean_episode_steps"].append(mean_episode_steps)
                Statistics["best_mean_episode_rewards"].append(best_mean_episode_reward)

                print("Episode %d" % (total_episodes,))
                print("reward: %f" % episode_reward)
                print("steps: %f" % episode_steps)
                print("best mean reward: %f" % best_mean_episode_reward)
                sys.stdout.flush()

                if total_episodes % self.log_steps == 0:
                    # Dump statistics to pickle
                    np.save(f'{self.reward_save_path}.npy', Statistics["mean_episode_rewards"])
                    np.save(f'{self.steps_save_path}.npy', Statistics["mean_episode_steps"])
                    print("Saved to %s" % 'training_reports')
                
                # reset tracking variables
                episode_steps = 0
                episode_reward = 0

            if done == False:
                I, z_critic, z_actor = self._update(ep_trajectory,
                                                    actor_bg,
                                                    critic_bg,
                                                    actor_bg_optimizer,
                                                    critic_bg_optimizer,
                                                    self.gamma,
                                                    I,
                                                    z_critic,
                                                    z_actor,
                                                    self.hid_dim)
    
    def _update(self,
                tuple, 
                actor, 
                value, 
                actor_optim, 
                critic_optim, 
                gamma, 
                I, 
                z_critic, 
                z_actor,
                hid_dim):

        lambda_critic = .5
        lambda_actor = .5

        
        state = torch.tensor([step[0] for step in tuple], device=self.device).unsqueeze(0)
        action = torch.tensor([step[1] for step in tuple], device=self.device).unsqueeze(0)
        reward = torch.tensor([step[2] for step in tuple], device=self.device).unsqueeze(0)
        next_state = torch.tensor([step[3] for step in tuple], device=self.device).unsqueeze(0)
        mask = torch.tensor([step[4] for step in tuple], device=self.device).unsqueeze(1)

        h_update = torch.zeros(size=(1, 1, hid_dim), device=self.device, dtype = torch.float32)
        x_update = torch.zeros(size=(1, 1, hid_dim), device=self.device, dtype = torch.float32)

        delta = reward + gamma * mask * value(next_state, h_update) - value(state, h_update)
        delta = delta.squeeze(0)[-1]

        # Critic Update
        critic_optim.zero_grad()
        z_critic_func = {}
        for param in z_critic:
            z_critic_func[param] = (gamma * lambda_critic * z_critic[param]).detach()
        critic_forward = value(state, h_update)
        critic_forward = torch.sum(critic_forward.squeeze())
        critic_forward.backward()
        # update z_critic and gradients
        for name, param in value.named_parameters():
            z_critic[name] = (z_critic_func[name] + param.grad).detach()
            param.grad = -delta.detach().squeeze() * (z_critic_func[name] + param.grad)

        # Actor Update
        actor_optim.zero_grad()
        z_actor_func = {}
        for param in z_actor:
            z_actor_func[param] = (gamma * lambda_actor * z_actor[param]).detach()
        _, log_prob, _, _, _, _, _ = actor.sample(state, h_update, x_update, sampling = False)
        log_prob = torch.sum(log_prob.squeeze())
        log_prob.backward()
        for name, param in actor.named_parameters():
            z_actor[name] = (z_actor_func[name] + I * param.grad).detach()
            param.grad = -delta.detach().squeeze() * (z_actor_func[name] + I * param.grad)

        I = gamma * I

        actor_optim.step()
        critic_optim.step()

        return I, z_critic, z_actor

        #this is a test