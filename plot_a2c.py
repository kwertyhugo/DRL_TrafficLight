import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
main_df = pd.read_csv('a2c_main_agent_history.csv')
sw_df   = pd.read_csv('a2c_sw_agent_history.csv')
se_df   = pd.read_csv('a2c_se_agent_history.csv')

# Function to plot metrics
def plot_agent_history(df, agent_name):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{agent_name} Training History', fontsize=16)
    
    # Rewards
    axs[0,0].plot(df['Episode'], df['Total_Reward'], color='green')
    axs[0,0].set_title('Total Reward')
    axs[0,0].set_xlabel('Episode')
    axs[0,0].set_ylabel('Reward')
    
    # Actor Loss
    axs[0,1].plot(df['Episode'], df['Actor_Loss'], color='blue')
    axs[0,1].set_title('Actor Loss')
    axs[0,1].set_xlabel('Episode')
    axs[0,1].set_ylabel('Loss')
    
    # Critic Loss
    axs[1,0].plot(df['Episode'], df['Critic_Loss'], color='red')
    axs[1,0].set_title('Critic Loss')
    axs[1,0].set_xlabel('Episode')
    axs[1,0].set_ylabel('Loss')
    
    # Entropy
    axs[1,1].plot(df['Episode'], df['Entropy'], color='orange')
    axs[1,1].set_title('Entropy')
    axs[1,1].set_xlabel('Episode')
    axs[1,1].set_ylabel('Entropy')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Plot all three agents
plot_agent_history(main_df, 'Main Intersection Agent')
plot_agent_history(sw_df, 'SW Ped Crossing Agent')
plot_agent_history(se_df, 'SE Ped Crossing Agent')
