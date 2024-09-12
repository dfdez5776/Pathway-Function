import numpy as np
import matplotlib.pyplot as plt


def average_reward_vis(reward_save_path, vis_save_path):  

    #load in steps and rewards, trying as a dict
    if type(reward_save_path) == str:
        data0 = np.load(reward_save_path, allow_pickle=True)
        data = data0.item()
    
    avg_steps = np.array(data.get('mean_episode_steps'))
    avg_reward = np.array(data.get('mean_episode_rewards'))
    steps = np.array(data.get('all_episode_steps'))
    reward = np.array(data.get('all_episode_rewards'))

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
        vis_save_path = f'{vis_save_path}_average.png'
        plt.savefig(vis_save_path)
     

def interval_reward_vis(reward_save_path, vis_save_path):

    if type(reward_save_path) == str:
        data0 = np.load(reward_save_path, allow_pickle=True)
        data = data0.item()
    
    steps = np.array(data.get('all_episode_steps'))
    reward_array = np.array(data.get('all_episode_rewards'))
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
        vis_save_path = f'{vis_save_path}_interval.png'
        plt.savefig(vis_save_path)
    
def gradient_vis(reward_save_path, vis_save_path):

    if type(reward_save_path) == str:
        data0 = np.load(reward_save_path, allow_pickle=True)
        data = data0.item()
    else:
        data = reward_save_path
    
    actor_gradients = data["actor_gradients"]
    critic_gradients = data["critic_gradients"]
    
    figure, subplot = plt.subplots(2)
    figure.suptitle('Gradients of Actor and Critic')
    for key, value in actor_gradients.items():
        subplot[0].plot(range(1, len(value) + 1), value, '.-', label = key)
    subplot[0].legend(fontsize = '4')
    subplot[0].set_title('Actor Gradients')
    for key, value in critic_gradients.items():
        subplot[1].plot(range(1, len(value) + 1), value, '.-', label = key)
    subplot[1].legend(fontsize = '4')
    subplot[1].set_title('Critic Gradients')
    
    if __name__ == "__main__":
        plt.show()
    else:  
        vis_save_path = f'{vis_save_path}_gradients.png'
        plt.savefig(vis_save_path)


def loss_vis(reward_save_path, vis_save_path):

    if type(reward_save_path) == str:
        data0 = np.load(reward_save_path, allow_pickle=True)
        data = data0.item()
    
    actor_loss = data["actor_loss"]
    critic_loss = data["critic_loss"]
    critic_target_loss = data["critic_target_loss"]
    sampled_entropies = data["sampled_entropies"]
    batch_entropies = data["batch_entropies"]
    alpha = data["alpha"]

    print(alpha)


    figure, subplot = plt.subplots(4)
    subplot[0].plot(actor_loss, label = 'actor loss')
    subplot[0].legend()
    subplot[1].plot(critic_loss, label = 'critic loss')
    subplot[1].legend()
    subplot[2].plot(sampled_entropies, label = 'sampled entropies')
    subplot[2].plot(batch_entropies, label = 'batch entropies')
    subplot[2].legend()
    subplot[3].plot(alpha, label = 'alpha')
    subplot[3].legend()

    if __name__ == "__main__":
        plt.show()
    else:
        vis_save_path0 = f'{vis_save_path}_loss.png'
        plt.savefig(vis_save_path0) 

def mean_std_vis(reward_save_path, vis_save_path):

    if type(reward_save_path ==  str):
        data0 = np.load(reward_save_path, allow_pickle=True)
        data  = data0.item()

    mean = data['mean']
    std = data['std']

    figure, subplot = plt.subplots(2)
    subplot[0].plot(mean, label = 'mean')
    subplot[0].legend()
    subplot[1].plot(std, label = 'std')
    subplot[1].legend()



        
    if __name__ == "__main__":
        plt.show()
    else:  
        vis_save_path = f'{vis_save_path}_mean_std.png'
        plt.savefig(vis_save_path)

    #load in data like before

    #make two subplots, one for mean and one for std


    





def activity_vis(reward_save_path, vis_save_path, activity_dict):

    print(len(activity_dict['d1 right reach']))
    

    figure, subplot = plt.subplots(2)
    figure.suptitle('D1 and D2 activity during Right reaches')
    subplot[0].plot(activity_dict['d1 right reach'], color = 'red', linewidth = .5)  
    subplot[0].set_title('D1')
    subplot[1].plot(activity_dict['d2 right reach'], color = 'blue', linewidth = .5)
    subplot[1].set_title('d2 right reach')

    
    plt.show()

    figure, subplot = plt.subplots(2)
    figure.suptitle('D1 and D2 activity during Left reach')
    subplot[0].plot(activity_dict['d1 left reach'], color = 'red', linewidth = .5)  
    subplot[0].set_title('D1')
    subplot[1].plot(activity_dict['d2 left reach'], color = 'blue', linewidth = .5)
    subplot[1].set_title('d2 left reach')

    
    plt.show()

    #activity = data["activity_magnitude"]
    #print(activity)
    
    #activity_vis = plt.figure()
    #plt.plot(range(1, len(activity)+1), activity)
    #plt.suptitle('Activity')

    
    #if __name__ == "__main__":
    #    plt.show()
    #else:
    #    vis_save_path = f'{vis_save_path}_activity.png'
    #    plt.savefig(vis_save_path)


def test_PSTH(str_activity, thal_activity, motor_activity, length_Rtrial, length_Ltrial): 


    plt.plot( str_activity["indirect_activityL"])
    plt.show()
    


    



#to run on local from saved paths
def main():


    #load in data

    reward_save_path = r'training_reports\07_normalization.npy'
   
    vis_save_path = r'visualizations\07_normalization_final'




    rewards = np.load(reward_save_path, allow_pickle=True)
    data = rewards.item()
    
    
    #visualize_steps_rewards(data, vis_save_path0)
    #interval_averages(data, vis_save_path1)
    gradient_vis(data, vis_save_path)

    #activity_vis(data,  r'training_reports\test.npy')



if __name__ == "__main__":
    main()