# ############################################
# Two link robot arm simulation environment
#
# Author : Deepak Raina @ IIT Delhi
# Version : 0.4
# ############################################

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from jax import config
config.update("jax_enable_x64", True)
import meshcat
import meshcat.geometry as geom
import meshcat.transformations as tf
import numpy as onp
import time
import jax.numpy as np
import jax
from jax import jacfwd, hessian
import numpy as np

#Environment class
class TwoLinkArmEnv(gym.Env):
    def __init__(self, reward_type = 0):

        self.target = np.array([0.0, 0.0]) #[x,y]
        self.state = np.array([0.0]*4) #[theta1, theta2, thetadot1, thetadot2]
        self.obs_state = None

        self.viewer = None 
        self._l1 = 1.0
        self._l2 = 1.0
        self._m1 = 0.5
        self._m2 = 0.5
        self._g = 9.81

        self.target_radius = 0.1
        self.max_speed = np.pi #rad/sec
        self.dt = 0.05 #time step
        self.max_time = 5
        self.step_n = 0
        self.reward_ver = reward_type
        
        # Target max-min limit
        self.target_high = [2, 2]
        self.target_low = [-1*i for i in self.target_high]
        # Joint max-min limit
        self.joint_high = [np.pi, np.pi]
        self.joint_low = [-1*i for i in self.joint_high]

        if self.reward_ver == 3:
            self.joint_low = [0.0, 0.0] #joint min limit 
            self.target_low = [-2, 0] #target min limit 

        #attributes
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32)
        self.observation_high = np.array(self.target_high*3 + [self.max_speed]*2)
        self.observation_space = spaces.Box(low=-self.observation_high, high=self.observation_high, dtype=np.float32)
        self.seed() #initialize a seed
    
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
        return self._m1 * self._g * self.p1(q)[1] + self._m2 * self._g * self.p2(q)[1]

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
        J_u = np.array([1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]) @ u

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
        hand_x, hand_y = self.__p1(self.state[:2]), self.__p2(self.state[:2])
        hand_pos = np.array([hand_x, hand_y])
        return 1 / 1000**(self.euclidian_distance(hand_pos, self.target))
    
    def done(self):
        hand_x, hand_y = self.__p1(self.state[:2]), self.__p2(self.state[:2])
        hand_pos = np.array([hand_x, hand_y])
        if self.euclidean_distance(hand_pos, self.target) < self.target_radius:
            return True
        # Also add max timestep termination
        return False  
        
    def step(self, action):
        self.state = self.__rk4_step(self.__f, self.state, self.dt, action)
        self.obs_state = np.append(self.target, self.state, dtype=np.float32)
        reward = self.reward()
        done = self.done()
        return self.obs_state, reward, done

    def render(self, mode='human',close=False):
        #close the viewer when needed
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        
        #first time render is called on a new viewer, it has to be initialized
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            
            #initialize viewer
            self.viewer = rendering.Viewer(700,700)
            self.viewer.set_bounds(-4.2, 4.2, -4.2, 4.2)
            
            #create target circle
            target = rendering.make_circle(self.target_radius)
            target.set_color(1,0,0)
            self.target_transform = rendering.Transform()
            target.add_attr(self.target_transform)
            self.viewer.add_geom(target)
            
            #create first arm segment
            link1 = rendering.make_capsule(self.arm_length,0.2)
            link1.set_color(0.5, 0.5, 0.5)
            self.link1_transform = rendering.Transform()
            link1.add_attr(self.link1_transform)
            self.viewer.add_geom(link1)
            
            #create first joint
            joint1 = rendering.make_circle(0.1)
            joint1.set_color(0, 0, 0)
            # joint1.add_attr(self.link1_transform)
            self.viewer.add_geom(joint1)

            #create second arm segment
            link2 = rendering.make_capsule(self.arm_length,0.2)
            link2.set_color(0.65, 0.65, 0.65)
            self.link2_transform = rendering.Transform()
            link2.add_attr(self.link2_transform)
            self.viewer.add_geom(link2)
            
            #create second joint
            joint2 = rendering.make_circle(0.1)
            joint2.set_color(0, 0, 1)
            joint2.add_attr(self.link2_transform)
            self.viewer.add_geom(joint2)
            
            #create end-effector circle
            end_effector = rendering.make_circle(0.1)
            end_effector.set_color(0,1,0)
            self.end_effector_transform = rendering.Transform()
            end_effector.add_attr(self.end_effector_transform)
            self.viewer.add_geom(end_effector)
            
        obs_state = self.get_obs()
        
        #set the viewer in the object according to current state
        self.link1_transform.set_rotation(self.state[0])
        self.link2_transform.set_translation(obs_state[2],obs_state[3])
        self.link2_transform.set_rotation(self.state[0] + self.state[1])
        self.end_effector_transform.set_translation(obs_state[4],obs_state[5])
        
        # if self.last_action is None:
        self.target_transform.set_translation(self.target[0],self.target[1])
            
        return self.viewer.render(return_rgb_array = mode == 'rgb_array')
    
    def euclidian_distance(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def close(self):
        # print('close')
        if self.viewer:
            self.viewer.close()
            self.viewer = None