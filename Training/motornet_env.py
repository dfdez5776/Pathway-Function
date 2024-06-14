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
from effector import Effector



#Environment class

class EffectorTwoLinkArmEnv(gym.Env):
    
    def __init__(self, max_timesteps, reward_type = 0):

        self.target = np.array([0.0, 0.5]) #[x,y]  
        self.state = np.array([0.0]*4) #[theta1, theta2, thetadot1, thetadot2]
        self.obs_state = None

        self.viewer = None 
       

        self.target_radius = 0.1
        self.max_speed = np.pi #rad/sec
        self.dt = 0.05 #time step
        self.max_timesteps = max_timesteps
        self.step_n = 0
        self.reward_ver = reward_type
        
        # Target max-min limit
        self.target_high = [2, 2]
        self.target_low = [-1*i for i in self.target_high]
        # Joint max-min limit
        self.joint_high = [np.pi, np.pi]
        self.joint_low = [-1*i for i in self.joint_high]
        self.current_hand_pos = None
        

        #attributes
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=onp.float32)
        self.observation_high = onp.array(self.target_high*3 + [self.max_speed]*2)
        self.observation_space = spaces.Box(low=-self.observation_high, high=self.observation_high, dtype=onp.float32)
        self.seed() #initialize a seed

        #Motor Net Effector

        self.two_link_arm = mn.effector.Effector(
            skeleton = mn.skeleton.TwoDofArm(),
            muscle = mn.muscle.RigidTendonHillMuscleThelen(),
            name = 'Effector',
            n_ministeps = 1, 
            timestep =  self.dt,
            integration_method = 'euler',
            damping = 0,
            )
        
        self.two_link_arm.add_muscle(
            path_fixation_body = [0., 1.],
            path_coordinates = [[-.15, .03], [0.94, 0.017]],
            name = 'pectoralis',
            max_isometric_force = 838,
            tendon_length = 0.039,
            optimal_muscle_length = .134,
            normalized_slack_muscle_length = 1.48)
        
        self.two_link_arm.add_muscle(
            path_fixation_body=[0., 0., 1.],
            path_coordinates=[[.14, 0.], [.05, -.00], [0.153, 0.]],
            name='deltoid',
            max_isometric_force=1207,
            tendon_length=.066,
            optimal_muscle_length=.140,
            normalized_slack_muscle_length=1.52)
        
        self.two_link_arm.add_muscle(
            path_fixation_body=[1., 1., 2.],
            path_coordinates=[[0.03, 0.], [0.138, -0.019], [-0.04, -0.017]],
            name='tricepslat',
            max_isometric_force=1549,
            tendon_length=.187,
            optimal_muscle_length=.093,
            normalized_slack_muscle_length=1.45)
        
        self.two_link_arm.add_muscle(
            path_fixation_body=[0., 2.],
            path_coordinates=[[-0.052, 0.033], [0.044, 0.001]],
            name='biceps',
            max_isometric_force=414,
            tendon_length=.204,
            optimal_muscle_length=.137,
            normalized_slack_muscle_length=1.5)
    
    
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reward(self):
        hand_pos = self.current_hand_pos
        
        
        return 1 / 1000**(self.euclidian_distance(hand_pos, self.target))
    
    def reset(self, episode):
        if episode % 2 == 0 :
            self.target = onp.array([0.0, 0.5]) #[x,y]
        else :
            self.target = onp.array([0.0, -0.25]) #[x,y]
        self.two_link_arm.reset()
        self.state = np.array([0.0]*4)
        self.obs_state = np.append(self.target, self.state)
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
        
    def step(self, episode_steps, action):

        action = action
        
        
        #Integrate and get state

        #Calling effector and integrating, default is Euler
        self.two_link_arm.step(action)    #this is giving a nonetype error which is strange since the effector class initializes states as none, need to look into this more

        #Effector returns states as dictionary of "joint", "cartesian", "muscle", "geometry", "fingertip"
        state_dict = self.two_link_arm.states

        #Extract cartestian coords                          #not sure if should extract coords or joint, need to look at shapes and output
        self.state = np.array(state_dict.get("coordinates"))  #now state in numpy array, fingertip = coords of 'finger'
        self.current_hand_pos = np.array(state_dict.get("fingertip").squeeze())
        # Get full state
        self.obs_state = np.append(self.target, self.state)
        # Get reward
        reward = self.reward()
        # Get done
        done = self.done(episode_steps)
        
        
        
        # Return environment variables
        return onp.array(self.obs_state, dtype=onp.float32), onp.array([reward],  dtype=onp.float32), onp.array([done], dtype = onp.float32)
    
    def euclidian_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def close(self):
       
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    