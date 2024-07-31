import motornet as mn
import os
import sys
import numpy as onp
import torch as th
from typing import Any
from IPython import get_ipython
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from jax import config
config.update("jax_enable_x64", True)
import numpy as onp
from numpy import pi
import time
import jax.numpy as np
import pygame
from pygame import gfxdraw
from effector import Effector , RigidTendonArm26


#Environment class
class EffectorTwoLinkArmEnv(gym.Env):
    
    def __init__(self, max_timesteps, render_mode, task_version):

        
        self.state = onp.array([0.0]*12) 
        self.joints = onp.array([0.0]*4) 
        self.obs_state = None

        self.viewer = None 

        self.target_radius = 0.1

        self.max_speed = onp.pi #rad/sec
        self.dt = 5e-2 #time step
        self.max_timesteps = max_timesteps
        self.step_n = 0
        self.task_version = task_version
        
        # Target max-min limit
        self.target_high = [2, 2]
        self.target_low = [-1*i for i in self.target_high]
        # Joint max-min limit
        self.joint_high = [onp.pi, onp.pi]
        self.joint_low = [-1*i for i in self.joint_high]
        self.current_hand_pos = onp.array([0.0 , 0.0])
        
        #attributes
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=onp.float32)
        self.observation_high = onp.array(self.target_high*3 + [self.max_speed]*2)
        self.observation_space = spaces.Box(low=-self.observation_high, high=self.observation_high, dtype=onp.float32)
        self.seed() #initialize a seed

        #Motor Net Effector
        self.two_link_arm = RigidTendonArm26(
            muscle = mn.muscle.RigidTendonHillMuscleThelen(),
            name = 'Effector',
            n_ministeps = 1, 
            timestep =  self.dt,
            integration_method = 'euler',
            damping = 0,
            )
        
        self._l1 = self.two_link_arm.skeleton.l1
        self._l2 = self.two_link_arm.skeleton.l2

        #Visualizations 
        #Pygame
        self.render_mode = render_mode
        self.SCREEN_DIM = 200
        self.screen = None
        self.clock = None

        


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reward(self, episode_steps, total_episodes):
              
        reward = 0
        euclidian_distance = self.euclidian_distance(self.current_hand_pos, self.target)
        penalty = 1 - (1 / 1000**euclidian_distance)

        if self.task_version == "delay_task":

            if total_episodes % 2 == 0 and episode_steps <= 20:

                if sum(self.activation) >= 0.1:   #maybe lower this
                    reward = -10
                else:
                    reward = 0  

            elif total_episodes % 2 == 0:  
                reward = -1e-2 * penalty
                


            if total_episodes % 2 == 1 and episode_steps <= 80:       
                if sum(self.activation) >= 0.1:
                    reward = -10
                else:
                    reward = 0  

            elif total_episodes % 2 == 1:
                reward = -1e-2 * penalty
              
        

        if self.task_version == "original":  
            reward = -1e-2 * penalty - 1e-3 * onp.sum(self.activation)
            if euclidian_distance <= self.target_radius:
                reward = 1

        return reward
    
    def reset(self, episode):
        
        
        if episode % 2 == 0 :
            self.target = onp.array([-0.2, 0.55]) #[x,y]
        else:
            self.target = onp.array([0.55, 0.55]) #[x,y]

        if self.task_version == "delay_task":
            self.targets_pos = onp.array([0.0, 0.0, 0.0, 0.0])

        
        self.two_link_arm.reset()

        state_dict = self.two_link_arm.states

        self.joints = onp.array(state_dict.get("joint")).squeeze() #import from effector module instead
        self.activation = onp.array(state_dict.get("activation"))

        if self.task_version == "delay_task":
           
            self.obs_state = onp.concatenate([self.targets_pos, self.joints, self.activation])

        if self.task_version == "original":
            self.obs_state = onp.concatenate([self.target, self.joints, self.activation])

        self.render_2(self.joints[:2])
        return onp.array(self.obs_state, dtype = onp.float32)

    def done(self, t):
        # Get hand position
        hand_pos = self.current_hand_pos
        # See if distance is out of range
        if self.euclidian_distance(hand_pos, self.target) < self.target_radius:
            return True
        # Terminate if at max timestep
        if t == self.max_timesteps:
            return True
        return False  
        
    def step(self, episode_steps, action, total_episodes):  ##pass in episode number
        

        if self.task_version == "delay_task":

            if total_episodes % 2 == 0  and episode_steps <= 20: #(even will be 20 second, left reaches)
                self.targets_pos ==  onp.array([0.0, 0.0, 0.0, 0.0])

            if total_episodes % 2 == 1  and episode_steps <= 80: #(odd will be 80 second, right reaches)
                self.targets_pos == onp.array([0.0, 0.0, 0.0, 0.0])

            else:
                self.targets_pos == onp.array([-0.2, 5.5, 0.2, 5.5])  #after delay, both targets known


            #Integrate and get state

        #Take step in environment and integrate, default is Euler
        self.two_link_arm.step(action)    

        #Effector returns states as dictionary of "joint", "cartesian", "muscle", "geometry", "fingertip", "activation"
        state_dict = self.two_link_arm.states

        #Extract states       
        self.joints = onp.array(state_dict.get("joint").squeeze())  
        self.activation = onp.array(state_dict.get("activation"))
       
       

        #Extract hand pos for reward
        self.current_hand_pos = onp.array(state_dict.get("fingertip").squeeze())
        
        #Concatenate targets, joint angles, joint position, and muscle activation into full state
        #for delay task modeled after experiment
        if self.task_version == "delay_task":
            self.obs_state = onp.concatenate([self.targets_pos, self.joints, self.activation])


        #for original task
        if self.task_version == "original":
            self.obs_state = onp.concatenate([self.target, self.joints, self.activation])


        # Get reward
        reward = self.reward(episode_steps, total_episodes)

        # Get done
        done = self.done(episode_steps)

        #Visualize
        self.render_2(self.joints[:2])
        
        # Return environment variables
        return onp.array(self.obs_state, dtype=onp.float32), onp.array([reward],  dtype=onp.float32), onp.array([done], dtype = onp.float32)
    

    def euclidian_distance(self, a, b):
        return onp.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)



    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None





    #Visualization and Pygame
    def __p1(self, q):

        
        """
            Position of the first mass
            Input: 
                q - joints 
            Output: 
                x_1 - position of mass 1 
        """
        x = self._l1 * onp.cos(q[0]) 
        y = self._l1 * onp.sin(q[0]) 

        return onp.asarray([x, y])
    
    def __p2(self, q):
        """
            Position of the second mass
            Input: 
                q - joints 
            Output: 
                x_2 - position of mass 2
        """
        x = self._l1 * onp.sin(q[0]) + self._l2 * onp.sin(q[0] + q[1])
        y = -self._l1 * onp.cos(q[0]) - self._l2 * onp.cos(q[0] + q[1])

        return onp.asarray([x, y])

    def render_2(self, q):

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.SCREEN_DIM, self.SCREEN_DIM)
                )
        
            else: 
                self.screen = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.SCREEN_DIM, self.SCREEN_DIM))
        surface.fill((255,255, 255))

        bound = self._l1 + self._l2 + 0.2  #default
        scale = self.SCREEN_DIM/(bound * 3)
        offset = self.SCREEN_DIM / 2

        p1 = self.__p1(q)*scale
        p2 = self.__p2(q)*scale

        xys = onp.array([[0,0] , p1, p2])
        thetas = [q[0] - pi / 2, q[0] + q[1] - pi / 2] 
        link_lengths = [self._l1*scale, self._l2*scale]

        pygame.draw.line(
            surface,
            start_pos=(-2.2*scale + offset, 1 * scale + offset),
            end_pos = (2.2 * scale + offset, 1 * scale + offset),
            color = (0,0,0),
        )

        pygame.draw.circle(
            surface,
            center = onp.ndarray.tolist(self.target*scale + offset), 
            radius = 2,
            color = (255,0,0)

        )

        finger_x = int(self.current_hand_pos[0]*scale + offset)
        finger_y = int(self.current_hand_pos[1]*scale + offset)

        elbow_x = int(p1[0] + offset)
        elbow_y = int(p1[1] + offset)

        shoulder_x = int(0+offset)
        shoulder_y = int(0+offset)

        pygame.draw.line(surface, (0, 204, 204), (shoulder_x, shoulder_y), (elbow_x, elbow_y))
        pygame.draw.line(surface, (0, 204, 204), (elbow_x, elbow_y), (finger_x, finger_y))

        gfxdraw.aacircle(surface, shoulder_x, shoulder_y, int(0.05 * scale), (204, 204, 0))
        gfxdraw.filled_circle(surface, shoulder_x, shoulder_y, int(0.05*scale), (204,204,0))


        gfxdraw.aacircle(surface, elbow_x, elbow_y, int(0.05 * scale), (204, 204, 0))
        gfxdraw.filled_circle(surface, elbow_x, elbow_y, int(0.05*scale), (204,204,0))

        gfxdraw.aacircle(surface, finger_x, finger_y, int(0.05 * scale), (204, 204, 0))
        gfxdraw.filled_circle(surface, finger_x, finger_y, int(0.05*scale), (204,204,0))
        
        surface = pygame.transform.flip(surface, False, True)
        self.screen.blit(surface, (0,0))

        if self.render_mode == "human" :
            pygame.event.pump()
            self.clock.tick(15) 
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return onp.transpose(
                onp.array(pygame.surfarray.pixels3d(self.screen)), axes = (1,0,2)
            )

    def euclidian_distance(self, a, b):
        return onp.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def close(self):
        # print('close')
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    