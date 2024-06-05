import gym
from gym import error, spaces, utils
from gym.utils import seeding
from jax import config
config.update("jax_enable_x64", True)
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as onp
from numpy import pi
import time
import jax.numpy as np
import jax
from jax import jacfwd, hessian
from arm_vis import Arm

#Environment class
class TwoLinkArmEnv(gym.Env):
    def __init__(self, max_timesteps, render_mode, reward_type = 0):

        self.target = np.array([0.0, 0.0]) #[x,y]
        self.state = np.array([0.0]*4) #[theta1, theta2, thetadot1, thetadot2]
        self.obs_state = None

        self.viewer = None 
        self._l1 = 1.0
        self._l2 = 1.0
        self._m1 = 0.1
        self._m2 = 0.1
        self._g = 9.81

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
        self.vis = Arm(self._l1, self._l2, self._m1, self._m2)

        #attributes
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=onp.float32)
        self.observation_high = onp.array(self.target_high*3 + [self.max_speed]*2)
        self.observation_space = spaces.Box(low=-self.observation_high, high=self.observation_high, dtype=onp.float32)
        self.seed() #initialize a seed
    
        #Visualization in Pygame
        self.render_mode = render_mode
        self.SCREEN_DIM = 200
        self.screen = None
        self.clock = None

    
    def __rk4_step(self, x, f, dt, *args):
        # one step of runge-kutta integration
        k1 = dt * f(x, *args)
        k2 = dt * f(x + k1/2, *args)
        k3 = dt * f(x + k2/2, *args)
        k4 = dt * f(x + k3, *args)
        return x + 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)
    
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
    
    def __get_jacfwd_pos(self):
        jac_p1 = jacfwd(self.__p1) 
        jac_p2 = jacfwd(self.__p2)
        return jac_p1, jac_p2

    def __KE_derived(self, q, qdot):
        jac_p1, jac_p2 = self.__get_jacfwd_pos()
        vels1 = jac_p1(q)@qdot
        vels2 = jac_p2(q)@qdot
        return 0.5 * self._m1 * (vels1[0]**2 + vels1[1]**2) + 0.5 * self._m2 *(vels2[0]**2 + vels2[1]**2)

    def __PE_derived(self, q):
        return self._m1 * self._g * self.__p1(q)[1] + self._m2 * self._g * self.__p2(q)[1]

    def __L_derived(self, q, qdot):
        return self.__KE_derived(q, qdot) - self.__PE_derived(q)

    def __get_lagrange(self):
        M_derived = hessian(self.__L_derived, argnums=1)
        C_derived = jacfwd(jacfwd(self.__L_derived, argnums=1))
        G_derived = jacfwd(self.__L_derived)
        return M_derived, C_derived, G_derived
    
    def __f(self, x, u):
        """
        Input: 
            state = x = [q, qdot] -- the state of the system 
        Output: 
            xdot = [qdot, qddot]
        """
        q, qdot = np.split(x, 2)

        # Only control joint position, not velocity
        J_u = np.eye(2) @ u.T
        J_u = J_u.squeeze()

        M_derived, C_derived, G_derived = self.__get_lagrange()
        M_inv = np.linalg.inv(M_derived(q, qdot))
        b = C_derived(q, qdot) @ qdot - G_derived(q, qdot)
        
        qddot = np.dot(
            M_inv, 
            J_u - b
        )

        xdot = np.hstack([qdot, qddot])
        return xdot 
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reward(self):
        hand_pos = self.__p2(self.state[:2])
        return 1 / 1000**(self.euclidian_distance(hand_pos, self.target))
    
    def reset(self, episode):
        if episode % 2 == 0 :
            self.target = np.array([1.0, 0.0]) #[x,y]
        else :
            self.target = np.array([-1.0, 0.0]) #[x,y]
        self.state = np.array([0.0]*4)
        self.obs_state = np.append(self.target, self.state)
        self.render_2(self.state[:2])
        return onp.array(self.obs_state, dtype = onp.float32)

    def done(self, t):
        # Get hand position
        hand_pos = self.__p2(self.state[:2])
        # See if distance is out of range
        if self.euclidian_distance(hand_pos, self.target) < self.target_radius:
            return True
        # Terminate if at max timestep
        if t == self.max_timesteps:
            return True
        return False  
        
    def step(self, episode_steps, action):
        # Integrate and get state
        self.state = self.__rk4_step(self.state, self.__f, self.dt, action)
        # Get full state
        self.obs_state = np.append(self.target, self.state)
        # Get reward
        reward = self.reward()
        # Get done
        done = self.done(episode_steps)
        # Visualize
        time.sleep(self.dt)
        self.vis.render(self.state[:2])
        #pygame visualizer
        self.render_2(self.state[:2])
        # Return environment variables
        return onp.array(self.obs_state, dtype=onp.float32), onp.array([reward],  dtype=onp.float32), onp.array([done], dtype = onp.float32)
    
    def render_2(self, q):
        import pygame
        from pygame import gfxdraw

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
        scale = self.SCREEN_DIM/(bound * 2)
        offset = self.SCREEN_DIM / 2

        p1 = self.__p1(q)*scale
        p2 = self.__p2(q)*scale

        xys = np.array([ p1, p2])
        thetas = [q[0], q[0] + q[1]] #not rlly sure why they subtract pi but we can remove later
        link_lengths = [self._l1*scale, self._l2*scale]

        pygame.draw.line(
            surface,
            start_pos=(-2.2*scale + offset, 1 * scale + offset),
            end_pos = (2.2 * scale + offset, 1 * scale + offset),
            color = (0,0,0),
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