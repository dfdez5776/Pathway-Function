import gym
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from agent_training import OptimizerSpec, On_Policy_Agent, Off_Policy_Agent
from optimization import optimizer 
from two_link_env import TwoLinkArmEnv
from motornet_env import EffectorTwoLinkArmEnv
import torch
import config
import os

def main():


    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    ### CREATE ENVIRONMENT ###
    torch.manual_seed(args.seed)
    env = EffectorTwoLinkArmEnv(args.max_timesteps, args.render_mode, args.task_version, args.test_train)

    ### OPTIMIZERS ###
    optimizer_spec_actor = OptimizerSpec(
        constructor=optim.AdamW,
        kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
    )

    optimizer_spec_critic = OptimizerSpec(
        constructor=optim.AdamW,
        kwargs=dict(lr=args.lr),
    )
    if args.algorithm == "optimization":
        print(type(args.algorithm))
        setup = optimizer(env,
                          args.max_episodes,
                          args.inp_dim,
                          args.hid_dim,
                          args.action_dim,
                          args.action_scale,
                          args.action_bias,
                          optimizer_spec_actor)

    if args.algorithm == "SAC":
        setup = Off_Policy_Agent(args.policy_replay_size,  
                                    args.policy_batch_size, 
                                    args.policy_batch_iters,
                                    args.lr,
                                    args.alpha,
                                    env, 
                                    args.seed,
                                    args.inp_dim,
                                    args.hid_dim,
                                    args.action_dim,
                                    optimizer_spec_actor,
                                    optimizer_spec_critic,
                                    args.tau,  
                                    args.gamma,
                                    args.save_iter,
                                    args.log_steps,
                                    args.frame_skips,
                                    args.model_save_path,
                                    args.buffer_save_path,
                                    args.reward_save_path,
                                    args.vis_save_path,
                                    args.action_scale,
                                    args.action_bias,
                                    args.automatic_entropy_tuning,
                                    args.continue_training,
                                    args.test_train )
    if args.algorithm == "SAC":
        setup = On_Policy_Agent(env,
                                args.seed,
                                args.inp_dim,
                                args.hid_dim,
                                args.action_dim,
                                optimizer_spec_actor,
                                optimizer_spec_critic,
                                args.gamma,
                                args.save_iter,
                                args.log_steps,
                                args.frame_skips,
                                args.model_save_path,
                                args.reward_save_path,
                                args.vis_save_path,
                                args.action_scale,
                                args.action_bias)
    
    
    if args.test_train == "test":
        setup.test(args.max_steps) 
    else:
        setup.train(args.max_steps, args.continue_training)

if __name__ == '__main__':
    main()
