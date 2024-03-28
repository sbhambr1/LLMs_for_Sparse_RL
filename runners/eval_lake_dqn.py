import pickle
import matplotlib.pyplot as plt
import numpy as np

# Open the pickle file in read mode
def load_pickle(file_path):
    with open(file_path, "rb") as file:
        # Load the data from the pickle file
        data = pickle.load(file)
        return data

def plot_rewards(vanilla_dqn, relabeled_random, relabeled_llm, map_size):
    
    vanilla_dqn = np.convolve(vanilla_dqn, np.ones(50)/50, mode='valid')
    relabeled_random = np.convolve(relabeled_random, np.ones(50)/50, mode='valid')
    relabeled_llm = np.convolve(relabeled_llm, np.ones(50)/50, mode='valid')
    
    plt.figure()
    plt.title("Rewards")
    plt.plot(vanilla_dqn, label='Vanilla DQN Reward', color='#F6CE3B', alpha=1)
    plt.plot(relabeled_random, label='Relabeled Random Reward', color='#FF5733', alpha=1)
    plt.plot(relabeled_llm, label='Relabeled LLM Reward', color='#33FF57', alpha=1)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend()
    
    plt.savefig(f'./runners/plots/reward/reward_comparison_plot_{map_size}x{map_size}_0.1_new_examples.png', format='png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close() 
    
def plot_losses(vanilla_dqn, relabeled_random, relabeled_llm, map_size):
        
    plt.figure()
    plt.title("Losses")
    plt.plot(vanilla_dqn, label='Vanilla DQN Loss', color='#F6CE3B', alpha=1)
    plt.plot(relabeled_random, label='Relabeled Random Loss', color='#FF5733', alpha=1)
    plt.plot(relabeled_llm, label='Relabeled LLM Loss', color='#33FF57', alpha=1)
    plt.xlabel("Episode")
    plt.ylabel("Losses")
    plt.legend()
    
    plt.savefig(f'./runners/plots/loss/loss_comparison_plot_{map_size}x{map_size}_0.1_new_examples.png', format='png', dpi=600, bbox_inches='tight')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    plt.clf()
    plt.close()

if __name__ == '__main__':
    
    map_size = 4 # Size of the map: 4x4, 8x8
    
    reward_path = './runners/plots/reward/'
    loss_path = './runners/plots/loss/'
    
    vanilla_dqn_reward_path = f'{reward_path}lake_4x4/reward_history.pkl'
    vanilla_dqn_loss_path = f'{loss_path}lake_4x4/loss_history.pkl'
        
    relabled_random_reward_path = f'{reward_path}lake_4x4_relabeled_random/reward_history.pkl'
    relabled_random_loss_path = f'{loss_path}lake_4x4_relabeled_random/loss_history.pkl'
    
    relabeled_llm_reward_path = f'{reward_path}lake_4x4_relabeled_llm_0.1_new_examples/reward_history.pkl'
    relabeled_llm_loss_path = f'{loss_path}lake_4x4_relabeled_llm_0.1_new_examples/loss_history.pkl'
    
    vanilla_dqn_reward = load_pickle(vanilla_dqn_reward_path)
    vanilla_dqn_loss = load_pickle(vanilla_dqn_loss_path)
    
    relabeled_random_reward = load_pickle(relabled_random_reward_path)
    relabeled_random_loss = load_pickle(relabled_random_loss_path)
    
    relabeled_llm_reward = load_pickle(relabeled_llm_reward_path)
    relabeled_llm_loss = load_pickle(relabeled_llm_loss_path)
    
    plot_rewards(vanilla_dqn_reward, relabeled_random_reward, relabeled_llm_reward, map_size)
    plot_losses(vanilla_dqn_loss, relabeled_random_loss, relabeled_llm_loss, map_size)
    