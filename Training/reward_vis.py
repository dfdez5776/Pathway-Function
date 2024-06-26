import numpy as np
import matplotlib.pyplot as plt

def main():
    visualize_steps_rewards()


def visualize_steps_rewards():  

    #load in steps and rewards, trying as a dict
    performance_dict = np.load(r'C:\Users\dfdez\OneDrive\Documents\GitHub\Pathway-Function\Training\training_reports\two_link_bg_reward2.npy', allow_pickle=True)

    avg_steps = np.array(performance_dict.item().get('mean_episode_steps'))
    avg_reward = np.array(performance_dict.item().get('mean_episode_rewards'))
    steps = np.array(performance_dict.item().get('all_episode_steps'))
    reward = np.array(performance_dict.item().get('all_episode_rewards'))
    reward = 100*reward #scaling for vis

    



    #avg_steps = np.load(r'C:\Users\dfdez\OneDrive\Documents\GitHub\Pathway-Function\Training\training_reports\two_link_bg_steps.npy')
    #avg_reward = np.load(r'C:\Users\dfdez\OneDrive\Documents\GitHub\Pathway-Function\Training\training_reports\two_link_bg_reward.npy')
    #steps =  np.load(#PATH)
    #reward = np.load(#PATH)
        
    
    #Make a simple x,y plot
    num_episodes = np.size(steps)+1
   
    x = np.arange(1, num_episodes)


    figure, subplot = plt.subplots(3)
    figure.suptitle('Steps and Rewards No Distance')
    subplot[0].plot(x, steps, color = 'red', linewidth = .5, label = 'Steps') 
    subplot[0].plot(x, reward, color = 'blue', linewidth = .5, label = 'Reward')
    subplot[0].set_title('Steps')
    subplot[0].legend(loc='upper right')
    subplot[1].plot(x, avg_steps, color = 'red', linewidth = .5)  
    subplot[1].set_title('Average Steps')
    subplot[2].plot(x, avg_reward, color = 'blue', linewidth = .5)
    subplot[2].set_title('Average Reward')

  

    plt.show()



