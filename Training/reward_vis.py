import numpy as np
import matplotlib.pyplot as plt


def visualize_steps_rewards(reward_save_path, vis_save_path):  

    #load in steps and rewards, trying as a dict
    performance_dict = np.load(reward_save_path, allow_pickle=True)
    
    avg_steps = np.array(performance_dict.item().get('mean_episode_steps'))
    avg_reward = np.array(performance_dict.item().get('mean_episode_rewards'))
    steps = np.array(performance_dict.item().get('all_episode_steps'))
    reward = np.array(performance_dict.item().get('all_episode_rewards'))

      
    
    #Plot of Average Rewards
    num_episodes = np.size(steps)+1
   
    x = np.arange(1, num_episodes)

    figure, subplot = plt.subplots(2)
    figure.suptitle('Steps and Rewards without Distance')
    subplot[0].plot(x, avg_steps, color = 'red', linewidth = .5)  
    subplot[0].set_title('Average Steps')
    subplot[1].plot(x, avg_reward, color = 'blue', linewidth = .5)
    subplot[1].set_title('Average Reward')

  
    
    
    if __name__ == "__main__":
        plt.show()
    else:
        plt.savefig(vis_save_path)
        
    




def interval_averages(reward_save_path, save_vis_path):



    performance_dict = np.load(reward_save_path, allow_pickle=True)
    steps = np.array(performance_dict.item().get('all_episode_steps'))
    reward_array = np.array(performance_dict.item().get('all_episode_rewards'))
    num_episodes = np.size(reward_array)
    

    #initialize counter and tuple of interval averages
    i = 0
    interval = 100 #size of intervals to take average over
    interval_averages_rew = []
    interval_averages_step = []

    while i < num_episodes:
        interval_rewards = reward_array[i: i + interval]
        interval_steps = steps[i: i + interval]
        interval_average_rewards = np.mean(interval_rewards) 
        interval_average_steps = np.mean(interval_steps)
        interval_averages_rew.append(interval_average_rewards)
        interval_averages_step.append(interval_average_steps)
        i += interval

    
    x0 = np.size(interval_averages_rew) + 1

    x = np.arange(1, x0)
    print(x)

    figure, subplot = plt.subplots(2)
    figure.suptitle('Average Rewards/Steps over Interval ( Sparse)')
    subplot[0].scatter(x, interval_averages_rew, color = 'black', linewidth = .5)
    subplot[0].plot(x, interval_averages_rew, color = 'red', linewidth = .5) 
    subplot[0].set_title('Average Reward over Interval')
    subplot[1].scatter(x, interval_averages_step, color = 'black', linewidth = .5)
    subplot[1].plot(x, interval_averages_step, color = 'red', linewidth = .5) 
    subplot[1].set_title('Average # Steps over Interval')
    plt.xlabel('Inverval # (n = 50 episodes)')

    if __name__ == "__main__":
        plt.show()
    else:
        plt.savefig(save_vis_path)
    
   

#to run on local from saved paths
def main():
    reward_save_path = r'C:\Users\dfdez\OneDrive\Documents\GitHub\Pathway-Function\Training\training_reports\two_link_bg_reward_sparse.npy'
    vis_save_path0 = r'C:\Users\dfdez\OneDrive\Documents\GitHub\Pathway-Function\Training\training_reports\visualization_sparse_average.png'
    vis_save_path1 = r'C:\Users\dfdez\OneDrive\Documents\GitHub\Pathway-Function\Training\training_reports\visualization_sparse.png'
    visualize_steps_rewards(reward_save_path, vis_save_path0)
    interval_averages(reward_save_path, vis_save_path1)


if __name__ == "__main__":
    main()