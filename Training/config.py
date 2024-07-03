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

    parser.add_argument('--reward_save_path', 
                        type=str, 
                        default='',
                        help='path to folder and file name to save rewards (do not put extension .npy)')

    parser.add_argument('--vis_save_path', 
                        type=str, 
                        help='path to folder and file name to save visualizations (do not put extension .npy)')
    
    parser.add_argument('--load_model_checkpoint', 
                        type=str, 
                        default="no",
                        help='load in checkpoint or not to continue training')
    return parser