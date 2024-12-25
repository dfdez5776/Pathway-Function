import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")

    # General Training Parameters
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.99, 
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0003, 
                        help='learning rate (default: 0.001)')

    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=0, 
                        help='weight decay parameter')

    parser.add_argument('--eps', 
                        type=float, 
                        default=0.01, 
                        help='optimizer epsilon')

    parser.add_argument('--seed', 
                        type=int, 
                        default=0, 
                        help='random seed (default: 123456)')

    parser.add_argument('--hid_dim', 
                        type=int, 
                        default=32, 
                        help='hidden size')

    parser.add_argument('--action_dim', 
                        type=int, 
                        default=1, 
                        help='action size')

    parser.add_argument('--inp_dim', 
                        type=int, 
                        default=3, 
                        help='state size')

    parser.add_argument('--log_steps', 
                        type=int, 
                        default=10, 
                        help='episodes before logging stats')

    parser.add_argument('--save_iter', 
                        type=int, 
                        default=10000, 
                        help='number of episodes until checkpoint is saved')

    parser.add_argument('--dt', 
                        type=float, 
                        default=0.001, 
                        help='dt of environment')

    parser.add_argument('--max_timesteps', 
                        type=int, 
                        default=100, 
                        help='number of timesteps for single episode (num / dt)')
    
    parser.add_argument('--max_episodes', 
                        type=int, 
                        default=6000, 
                        help='maximum episodes to run')
    
    parser.add_argument('--render_mode', 
                        type=str, 
                        default="human", 
                        help='human or rgb_array for visualization in pygame. Human creates a display')

    parser.add_argument('--frame_skips', 
                        type=int, 
                        default=2,
                        help='number of times to repeat same action')

    parser.add_argument('--action_scale', 
                        type=float, 
                        default=0.5, 
                        help='scale of actor action')

    parser.add_argument('--action_bias', 
                        type=float, 
                        default=0.5, 
                        help='bias of actor action')

    parser.add_argument('--max_steps', 
                        type=int, 
                        default=1000000,
                        help='maximum number of steps to use for training')
    
    # Saving Parameters
    parser.add_argument('--model_save_path', 
                        type=str, 
                        default='',
                        help='path to folder and file name of model to save (do not put extension pth)')
    parser.add_argument('--buffer_save_path', 
                        type=str, 
                        default='',
                        help='path to save buffer replay list')
    parser.add_argument('--reward_save_path', 
                        type=str, 
                        default='',
                        help='path to folder and file name to save rewards (do not put extension .npy)')

    parser.add_argument('--vis_save_path', 
                        type=str, 
                        help='path to folder and file name to save visualizations (do not put extension .npy)')
    
    parser.add_argument('--test_train', 
                        type=str, 
                        default="no",
                        help='load in checkpoint or not to continue training')
    
    parser.add_argument('--continue_training', 
                        type=str, 
                        default="no", 
                        help='option to continue training a previous model')
    
    parser.add_argument('--algorithm', 
                        type=str, 
                        default="SAC", 
                        help='which algorithm to use. SAC, Actor-Critic w Eligibility trace, Optimization')
    parser.add_argument('--policy_replay_size', 
                        type=int, 
                        default=4000, 
                        help='size of replay buffer for SAC')
    parser.add_argument('--policy_batch_size', 
                        type=int, 
                        default=8, 
                        help='Size of sample from replay memory to update')
    parser.add_argument('--policy_batch_iters', 
                        type=int, 
                        default=1, 
                        help='how many time to repeat replay step')
    parser.add_argument('--tau', 
                        type=float, 
                        default=0.005, 
                        help='constant for critic update')
    parser.add_argument('--automatic_entropy_tuning', 
                        type=bool, 
                        default=True, 
                        help='maximize entropy and include loss for it')
    parser.add_argument('--alpha', 
                        type= float, 
                        default= 0.2, 
                        help='SAC critic loss constant')
    
    parser.add_argument('--task_version', 
                        type=str, 
                        default= "original", 
                        help='different versions of task for arm environment. Delay_task modeled after Li et al paper')
    
    return parser