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
#from replay_buffer.py import PolicyReplayMemory

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
                vis_save_path,
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
        self.vis_save_path = vis_save_path
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _Normalize_Data(self, data, min, max):
        '''
            Mainly used for neural data if model is constrained
        '''
        return (data - min) / (max - min)

    def select_action(self, policy, state, hn, evaluate):
        '''
            Selection of action from policy, consistent across training methods
        '''
        
        
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

       
        
        hn = hn.to(self.device)


        if evaluate == False: 
            action, _, _, _, hn = policy.sample(state, hn, sampling=True)
        else:
            _, _, action, _, hn = policy.sample(state, hn, sampling=True)

        return action.detach().cpu().numpy(), hn.detach()
    

    def test(self, max_steps):
        
        checkpoint = torch.load(f'{self.model_save_path}.pth', map_location = torch.device('cpu'))
        actor_bg = RNN_MultiRegional(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias, self.device).to(self.device)
        actor_bg.load_state_dict(checkpoint['agent_state_dict'])
        #critic_bg = RNN.load_state_dict(checkpoint['critic_state_dict'])
      
        iteration = checkpoint['iteration']
        iteration0 = iteration

        str_activity = {"str_activityR" : np.empty([1,256]),
                    "str_activityL" : np.empty([1,256]),
                    "direct_activityL" : [] ,
                    "direct_activityR" : [], #change these to np arrays so we can take mean, just not sure dimensions
                    "indirect_activityL" : [],
                    "indirect_activityR" : []
                    }
        
        thal_activity = {"thal_activityR" : np.empty([1,256]),
                     "thal_activityL" : np.empty([1,256]),
                     #maybe record activity of thal into motor , not sure if all thal neurons input to motor
                    }
        

        motor_activity = {"motor_activityR" : np.empty([1,256]),
                      "motor_activityL" : np.empty([1,256])
                     }
        
        muscle_activation = {"muscle1_activation" : np.empty([1]),
                         "muscle2_activation" : np.empty([1]),
                         "muscle3_activation" : np.empty([1]),
                         "muscle4_activation" : np.empty([1]),
                         "muscle5_activation" : np.empty([1]),
                         "muscle6_activation" : np.empty([1]),
                     }
      
        #record activity of each muscle and make a sort of animation (maybe using pygame?)) that shows an anatomical model of the muscles.
        #or maybe add to existing pygame vis. 
       
        

        #indeces for extracting activity
        str_idx0 = 0
        str_idx1 = self.hid_dim

        thal_idx0 = str_idx1 
        thal_idx1 = thal_idx0 + self.hid_dim  

        motor_idx0 = thal_idx1 
        motor_idx1 = motor_idx0 + self.hid_dim




        #Initializing...
        state = self.env.reset(iteration)
        h_prev = torch.zeros(size=(1 ,1 , self.hid_dim*3), device = self.device)
        num_episodes = 0
        episode_steps = 0 


        #run episode as usual in train but without the update
        while iteration < max_steps:
         
           
            action, h_current = self.select_action(actor_bg, state, h_prev, evaluate = True)
            


            for _ in range(self.frame_skips):

                next_state, reward, done = self.env.step(episode_steps, action)
                
                if iteration == iteration0 or iteration == iteration0 + 1:  #activity for 1 episode

                    if iteration % 2 == 0: #left reach, even trials
                        

                        str_activity0 = np.array(h_current[: ,: , :str_idx1].squeeze(1))
                        tot_str_activityL = str_activity['str_activityL']
                        tot_str_activityL = np.append(tot_str_activityL, str_activity0, axis = 0)

                        #direct_activity0 = np.array(h_current[: ,: ,str_idx1/2].squeeze(1)) #add in all the d1 d2 stuff for left and right


                        thal_activity0 = np.array(h_current[: ,:, thal_idx0:thal_idx1].squeeze(1))
                        tot_thal_activityL = thal_activity['thal_activityL']
                        tot_thal_activityL = np.append(tot_thal_activityL, thal_activity0, axis = 0)
                    
                        motor_activity0 = np.array(h_current[: ,:, motor_idx0:].squeeze(1))
                        tot_motor_activityL = motor_activity['motor_activityL']
                        tot_motor_activityL = np.append(tot_motor_activityL, motor_activity0, axis = 0)
                    
                    else: #Right reach, odd trials


                        str_activity0 = np.array(h_current[: ,: , :str_idx1].squeeze(1))
                        tot_str_activityR = str_activity['str_activityR']
                        tot_str_activityR = np.append(tot_str_activityR, str_activity0, axis = 0)

                        thal_activity0 = np.array(h_current[: ,:, thal_idx0:thal_idx1].squeeze(1))
                        tot_thal_activityR = thal_activity['thal_activityR']
                        tot_thal_activityR = np.append(tot_thal_activityR, thal_activity0, axis = 0)
                    
                        motor_activity0 = np.array(h_current[: ,:, motor_idx0:].squeeze(1))
                        tot_motor_activityR = motor_activity['motor_activityR']
                        tot_motor_activityR = np.append(tot_motor_activityR, motor_activity0, axis = 0)


                        str_activity["str_activityR"] = tot_str_activityR
                        thal_activity["thal_activityR"] = tot_thal_activityR
                        motor_activity["motor_activityR"] = tot_motor_activityR

                
            

            #save activity of direct/indirect pathways for left/right trails 
                if done == False:
                    episode_steps += 1
                if done == True:
                    break
            
            state = next_state
            h_prev = h_current
        

            if done == True:

                if iteration % 2 == 0 and iteration == iteration0 or iteration == iteration0 + 1:
                    str_activity["str_activityL"] = np.mean(tot_str_activityL, axis = 1)
                    thal_activity["thal_activityL"] = np.mean(tot_thal_activityL, axis = 1)
                    motor_activity["motor_activityL"] = np.mean(tot_motor_activityL, axis = 1)
                

                elif iteration % 2 == 1 and iteration == iteration == iteration0 or iteration == iteration0 + 1: 
                    str_activity["str_activityR"] = np.mean(tot_str_activityR, axis = 1)
                    thal_activity["thal_activityR"] = np.mean(tot_thal_activityR, axis = 1)
                    motor_activity["motor_activityR"] = np.mean(tot_motor_activityR, axis = 1)

                
                
                iteration += 1
                episode_steps = 0
                state = self.env.reset(iteration)

                h_prev = torch.zeros(size=(1, 1, self.hid_dim * 3), device=self.device)
                




    
    def train(self, max_steps, continue_training):

        '''
            Train the agent using one step actor critic
        '''

        actor_bg = RNN_MultiRegional(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias, self.device).to(self.device)
        critic_bg = RNN(self.inp_dim, self.hid_dim).to(self.device)

        actor_bg_optimizer = self.optimizer_spec_actor.constructor(actor_bg.parameters(), **self.optimizer_spec_actor.kwargs)
        critic_bg_optimizer = self.optimizer_spec_critic.constructor(critic_bg.parameters(), **self.optimizer_spec_critic.kwargs)

        #option for loading in model
         #add new arg 
        if continue_training == "yes":

            
            checkpoint = torch.load(f'{self.model_save_path}.pth', map_location = torch.device('cpu'))

            #load in models
            actor_bg.load_state_dict(checkpoint['agent_state_dict'])
            critic_bg = critic_bg.load_state_dict(checkpoint['critic_state_dict'])

            #load in optimizers 
            actor_bg_optimizer.load_state_dict(checkpoint['agent_optimizer_state_dict'])
            critic_bg_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        

        z_actor = {}
        z_critic = {}
        grad_vis_actor = {}
        grad_vis_critic = {}
        I = 1
        for name, params in actor_bg.named_parameters():
            z_actor[name] = torch.zeros_like(params)
        for name, params in critic_bg.named_parameters():
            z_critic[name] = torch.zeros_like(params)
        for name, params in actor_bg.named_parameters():
            grad_vis_actor[name] = []
        for name, params in critic_bg.named_parameters():
            grad_vis_critic[name] = []
        
        Statistics = {
            "mean_episode_rewards": [],
            "mean_episode_steps": [],
            "best_mean_episode_rewards": [],
            "all_episode_steps":[],
            "all_episode_rewards":[],
            "actor_gradients": {},
            "critic_gradients": {},
            "activity_magnitude": []
        }

        episode_reward = 0
        sum_ep_rew = []
        best_mean_episode_reward = -float("inf")
        episode_steps = 0
        total_episodes = 0
        all_reward = []
        all_steps = []
      

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset(0)
        ep_trajectory = []

            #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, self.hid_dim * 3), device=self.device)
      




        ### STEPS PER EPISODE ###
        for t in range(max_steps):
            
            
            with torch.no_grad():
                action, h_current = self.select_action(actor_bg, state, h_prev, evaluate=False)  # Sample action from policy
            
            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(self.frame_skips):
                next_state, reward, done = self.env.step(episode_steps, action)
                if done == False:
                    episode_steps += 1
                episode_reward += reward[0]
                sum_ep_rew.append(reward[0])
                if done == True:
                    break

            mask = 1.0 if episode_steps == self.env.max_timesteps else float(not done)

            ep_trajectory.append((state, action, reward, next_state, mask))

            state = next_state
            h_prev = h_current

           

            I, z_critic, z_actor, grad_vis_critic, grad_vis_actor = self._update(ep_trajectory,    #actor_grad, critic_grad
                                                    actor_bg,
                                                    critic_bg,
                                                    actor_bg_optimizer,
                                                    critic_bg_optimizer,
                                                    self.gamma,
                                                    I,
                                                    z_critic,
                                                    z_actor,
                                                    grad_vis_critic,
                                                    grad_vis_actor,
                                                    done,
                                                    self.hid_dim)

            




            ### EARLY TERMINATION OF EPISODE
            if done:
              
                total_episodes += 1

                # Add stats to lists
                all_steps.append(episode_steps)
                all_reward.append(episode_reward)

                
             

                activity_norm = float(torch.norm(h_prev))
                
                

                # reset training conditions
                h_prev = torch.zeros(size=(1, 1, self.hid_dim * 3), device=self.device)
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
                if len(all_reward) > 0:
                    mean_episode_reward = np.mean(np.array(all_reward)[-1000:])
                if len(all_steps) > 0:
                    mean_episode_steps = np.mean(np.array(all_steps)[-1000:])
                if len(all_reward) > 10:
                    if mean_episode_reward >= best_mean_episode_reward: #save model if improving
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
                Statistics["all_episode_rewards"] = all_reward
                Statistics["all_episode_steps"] = all_steps
                Statistics["actor_gradients"] = grad_vis_actor
                Statistics["critic_gradients"] = grad_vis_critic
                Statistics["activity_magnitude"].append(activity_norm)

                

                print("Episode %d" % (total_episodes,))
                print("reward: %f" % episode_reward)
                print("steps: %f" % episode_steps)
                print("best mean reward: %f" % best_mean_episode_reward)
                sys.stdout.flush()

                if total_episodes % self.log_steps == 0:
                    # Dump statistics to pickle
                    np.save(f'{self.reward_save_path}.npy', Statistics)
                    print("Saved to %s" % 'training_reports')

                if total_episodes % 3000 == 0: #save graphs every 3000 episodes
                    average_reward_vis(f'{self.reward_save_path}.npy', self.vis_save_path)
                    interval_reward_vis(f'{self.reward_save_path}.npy', self.vis_save_path)
                    gradient_vis(f'{self.reward_save_path}.npy', self.vis_save_path)
                    activity_vis(f'{self.reward_save_path}.npy', self.vis_save_path)

                
                # reset tracking variables
                episode_steps = 0
                episode_reward = 0
    
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
                grad_vis_critic,
                grad_vis_actor,
                done,
                hid_dim):

        lambda_critic = .9
        lambda_actor = .9
        
        state = torch.tensor(np.array([step[0] for step in tuple]), device=self.device).unsqueeze(0)
        action = torch.tensor(np.array([step[1] for step in tuple]), device=self.device).unsqueeze(0)
        reward = torch.tensor(np.array([step[2] for step in tuple]), device=self.device).unsqueeze(0)
        next_state = torch.tensor(np.array([step[3] for step in tuple]), device=self.device).unsqueeze(0)
        mask = torch.tensor(np.array([step[4] for step in tuple]), device=self.device).unsqueeze(1)

        h_update_actor = torch.zeros(size=(1, 1, hid_dim*3), device=self.device, dtype = torch.float32)


        h_update_critic = torch.zeros(size=(1, 1, hid_dim), device=self.device, dtype = torch.float32)

        delta = reward + gamma * mask * value(next_state, h_update_critic) - value(state, h_update_critic)
        delta = delta.squeeze(0)[-1]

        # Critic Update
        critic_optim.zero_grad()
        z_critic_func = {}
        for param in z_critic:
            z_critic_func[param] = (gamma * lambda_critic * z_critic[param]).detach() 
        critic_forward = value(state, h_update_critic)
        
        if critic_forward.shape[1] > 1:
            critic_forward = critic_forward.squeeze()
            critic_forward = torch.mean(critic_forward)
        
        else:
            critic_forward = critic_forward.squeeze()

        critic_forward.backward()

        # update z_critic and gradients
        for name, param in value.named_parameters():
            z_critic[name] = (z_critic_func[name] + param.grad).detach()
            param.grad += z_critic_func[name]
            param.grad *= -delta.detach().squeeze()           
            if done == True:
                grad_norm = float(torch.norm(param.grad))
                grad_vis_critic[name].append(grad_norm)  



                    

        # Actor Update
        actor_optim.zero_grad()
        z_actor_func = {}
        for param in z_actor:
            z_actor_func[param] = (gamma * lambda_actor * z_actor[param]).detach()
        _, log_prob, _, _, _ = actor.sample(state, h_update_actor, sampling = False)
        log_prob = torch.mean(log_prob.squeeze())
        log_prob.backward()
        for name, param in actor.named_parameters():
            z_actor[name] = (z_actor_func[name] + I * param.grad).detach()
            param.grad *= I
            param.grad += z_actor_func[name]
            param.grad *= -delta.detach().squeeze()
            grad_norm = float(torch.norm(param.grad))
            if done == True:
                grad_vis_actor[name].append(grad_norm)  
                



        
           
        
        I = gamma * I

        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_optim.step()
    
        nn.utils.clip_grad_norm_(value.parameters(), 1.0)
        critic_optim.step()

        return I, z_critic, z_actor, grad_vis_critic, grad_vis_actor
    
class Off_Policy_Agent():
    def __init__(self, 
                 policy_replay_size,
                 policy_batch_size, 
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
                 reward_save_path,
                 vis_save_path,
                 action_scale,
                 action_bias,
                 automatic_entropy_tuning):
        

        self.policy_replay_size = policy_replay_size
        self.policy_batch_size = policy_batch_size
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
        self.reward_save_path = reward_save_path
        self.vis_save_path = vis_save_path
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        #self.policy_memory = PolicyReplayMemory

        #initialize Actor/Critic RNNs 

        self.actor = RNN_MultiRegional(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias)

        self.critic = RNN(self.inp_dim, self.hid_dim).to(self.device)
        self.critic_target = RNN(self.inp_dim, self.hid_dim).to(self.device)
        

        #Optimizers

        self.actor_optimizer = self.optimizer_spec_actor.constructor(self.actor.parameters(), **self.optimizer_spec_actor.kwargs)
        self.critic_optimizer = self.optimizer_spec_actor.constructor(self.critic.parameters(), **self.optimizer_spec_critic.kwargs)
        #hard_update(self.critic_target, self_critic)
        self.automatic_entropy_tuning = automatic_entropy_tuning
        

    

    #def policy replay (probably a separate file)

        

    def select_action(self, h_activity, x_activity, evaluate):
        
        #make sure state is a tensor, and send activity to gpu/cpu
        state = torch.tensor(state, dtype = torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

        h_activity = h_activity.to(self.device)
        x_activity = x_activity.to(self.device)


        if evaluate == False:
            action, _, _, _, h_activity, x_activity, _ = self.actor.sample(state, h_activity, x_activity, sampling = True)
        else:
            _, _, action, _, h_activity, x_activity, _ = self.actor.sample(state, h_activity, x_activity, sampling = True)
          ###review and check for edits in models: sampling and forward



        return action.detach().cpu().numpy(), h_activity.detach(), x_activity.detach()

    def train(self, max_steps, load_model_checkpoint):

        #Initialize tracking statistics 
        Statistics = {'best_mean_episode_rewards' : [],
                     'mean_episode_rewards' : [],
                     'mean_episode_steps' : [],
                     'all_episode_steps' : [],
                     'all_episode_rewards' : [],
                     'actor_gradients' : [],
                     'critic_gradients' : [],
                     'activity' : [],
                     }

        total_episodes = 0
        best_mean_episode_reward = -float("inf")
        all_steps = []
        all_reward = []
        actor_losses = []
        critic_losses = []
        ep_trajectory = []
        state = self.env.reset(0)

        h_prev = torch.zeros_like(size = (1 ,1 , self.hid_dim*3), device = self.device )
        x_prev = torch.zeros_like(size = (1, 1, self.hid_dim*3), device = self.device)

        done = False


        #Episode Training Loop
        for t in range(max_steps):

            episode_steps = 0
            episode_reward = 0

            #select action
            with torch.no_grad():
                action, h_current, x_current = self.select_action(state, h_prev, x_prev, evaluate = False)
            
            if len(self.policy_memory.buffer) > self.policy_batch_size:
                for _ in range(self.policy_batch_size):
                    critic_loss, critic_target_loss, policy_loss, grad_vis_actor, grad_vis_critic = self.update(ep_trajectory,
                                                                               done)
                    critic_losses.append(critic_loss)
                    actor_losses.append(policy_loss)


            for _ in range(self.frame_skips):
                next_state, reward, done = self.env.step(episode_steps, action)
                if done == False:
                    episode_steps += 1
                episode_reward += reward[0]
                if done == True:
                    break

                
            mask = 1.0 if episode_steps == self.env.max_timesteps else float(not done)

            ep_trajectory.append(state, action, reward, next_state, mask)

            state = next_state

            if done:

                total_episodes += 1
                all_steps.append(episode_steps)
                all_reward.append(episode_reward)

                activity_norm = float(torch.norm(h_prev))


                self.policy_memory.push(ep_trajectory)




                ### 4. Log progress and keep track of statistics
                if len(all_reward) > 0:
                    mean_episode_reward = np.mean(np.array(all_reward)[-1000:])
                if len(all_steps) > 0:
                    mean_episode_steps = np.mean(np.array(all_steps)[-1000:])
                if len(all_reward) > 10:
                    if total_episodes % 500 == 0: #save params every 500 episodes
                        torch.save({
                            'iteration': t,
                            'agent_state_dict': self.actor.state_dict(),
                            'critic_state_dict': self.critic.state_dict(),
                            'agent_optimizer_state_dict': self.actor_optimizer.state_dict(),
                            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                        }, self.model_save_path + '.pth')

                    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                Statistics["mean_episode_rewards"].append(mean_episode_reward)
                Statistics["mean_episode_steps"].append(mean_episode_steps)
                Statistics["best_mean_episode_rewards"].append(best_mean_episode_reward)
                Statistics["all_episode_rewards"] = all_reward
                Statistics["all_episode_steps"] = all_steps
                Statistics["actor_gradients"] = grad_vis_actor
                Statistics["critic_gradients"] = grad_vis_critic
                Statistics["activity_magnitude"].append(activity_norm)

                

                print("Episode %d" % (total_episodes,))
                print("reward: %f" % episode_reward)
                print("steps: %f" % episode_steps)
                print("best mean reward: %f" % best_mean_episode_reward)
                sys.stdout.flush()

                if total_episodes % self.log_steps == 0:
                    # Dump statistics to pickle
                    np.save(f'{self.reward_save_path}.npy', Statistics)
                    print("Saved to %s" % 'training_reports')

                if total_episodes % 3000 == 0: #save graphs every 3000 episodes
                    average_reward_vis(f'{self.reward_save_path}.npy', self.vis_save_path)
                    interval_reward_vis(f'{self.reward_save_path}.npy', self.vis_save_path)
                    gradient_vis(f'{self.reward_save_path}.npy', self.vis_save_path)
                    activity_vis(f'{self.reward_save_path}.npy', self.vis_save_path)


                #Reset lists and activity
                ep_trajectory = []
                episode_reward = []
                episode_steps = []

                h_prev = torch.zeros_like(size = (1 ,1, self.hid_dim*3), device = self.device)
                x_next =  torch.zeros_like(size = (1 ,1, self.hid_dim*3), device = self.device)

                state = self.env.reset(total_episodes)


                    

                    



    #def update

        #def hard update

        #def soft update

        #def automatic entropy tuning 


        #def loss functions





