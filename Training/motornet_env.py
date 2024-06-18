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

#Environment class
class EffectorTwoLinkArmEnv(gym.Env):
    
    def __init__(self, max_timesteps, render_mode, reward_type = 0):

        self.target = np.array([0.55, -0.3]) #[x,y]  
        self.state = np.array([0.0]*4) 
        self.joints = np.array([0.0]*4) 
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
        self.two_link_arm = mn.effector.RigidTendonArm26(
            muscle = mn.muscle.RigidTendonHillMuscleThelen(),
            name = 'Effector',
            n_ministeps = 1, 
            timestep =  self.dt,
            integration_method = 'euler',
            damping = 0,
            )
        
        self._l1 = self.two_link_arm.skeleton.l1
        self._l2 = self.two_link_arm.skeleton.l2

        #Visualization in Pygame
        self.render_mode = render_mode
        self.SCREEN_DIM = 200
        self.screen = None
        self.clock = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reward(self):
        hand_pos = self.current_hand_pos
        return 1 / 1000**(self.euclidian_distance(hand_pos, self.target))
    
    def reset(self, episode):
        if episode % 2 == 0 :
            self.target = onp.array([0.4, 0.4]) #[x,y]
        else :
            self.target = onp.array([0.0, 0.3]) #[x,y]
        self.two_link_arm.reset()
        self.state = np.array([0.0]*4)
        self.obs_state = np.append(self.target, self.state)
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
        
    def step(self, episode_steps, action):

        action = action
        #Integrate and get state

        #Calling effector and integrating, default is Euler
        self.two_link_arm.step(action)    

        #Effector returns states as dictionary of "joint", "cartesian", "muscle", "geometry", "fingertip"
        state_dict = self.two_link_arm.states

        #Extract cartestian coords          
        self.joints = onp.array(state_dict.get("joint").squeeze())               
        
        self.state = np.array(state_dict.get("cartesian")) 
        self.current_hand_pos = np.array(state_dict.get("fingertip").squeeze())
        # Get full state
        self.obs_state = np.append(self.target, self.state)
        # Get reward
        reward = self.reward()
        # Get done
        done = self.done(episode_steps)

        self.render_2(self.joints[:2])
        
        # Return environment variables
        return onp.array(self.obs_state, dtype=onp.float32), onp.array([reward],  dtype=onp.float32), onp.array([done], dtype = onp.float32)
    
    def euclidian_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

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
        x = self._l1 * np.sin(q[0])
        y = -self._l1 * np.cos(q[0])

        return np.asarray([x, y])
    
    def __p2(self, q):
        """
            Position of the second mass
            Input: 
                q - joints 
            Output: 
                x_2 - position of mass 2
        """
        x = self._l1 * np.sin(q[0]) + self._l2 * np.sin(q[0] + q[1])
        y = -self._l1 * np.cos(q[0]) - self._l2 * np.cos(q[0] + q[1])

        return np.asarray([x, y])

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
        scale = self.SCREEN_DIM/(bound * 4)
        offset = self.SCREEN_DIM / 2

        p1 = self.__p1(q)*scale
        p2 = self.__p2(q)*scale

        xys = np.array([[0,0] , p1, p2])
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

        for ((x,y), th, llen) in zip(xys, thetas, link_lengths):
            x = x + offset
            y = y + offset
            l, r, t, b = 0, llen, 0.1*scale, -0.1*scale
            coords = [(l,b), (l,t), (r,t), (r,b)]
            transformed_coords = []
            for coord in coords:
                coord = pygame.math.Vector2(coord).rotate_rad(th)
                coord = (coord[0] + x, coord[1] + y)
                transformed_coords.append(coord)
            gfxdraw.aapolygon(surface, transformed_coords, (0, 204, 204))
            gfxdraw.filled_polygon(surface, transformed_coords, (0, 204, 204))

            gfxdraw.aacircle(surface, int(x), int(y), int(0.1 * scale), (204, 204, 0))
            gfxdraw.filled_circle(surface, int(x), int(y), int(0.1*scale), (204,204,0))
        
        surface = pygame.transform.flip(surface, False, True)
        self.screen.blit(surface, (0,0))

        if self.render_mode == "human" :
            pygame.event.pump()
            self.clock.tick(15) #make this an arg in configs later
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes = (1,0,2)
            )

    def euclidian_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def close(self):
        # print('close')
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    