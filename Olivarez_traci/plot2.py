import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
# Point this to the folder where your BASELINE CSV is
INPUT_DIR = './Olivarez_traci/output_A2C_Baseline/'
CSV_FILE = 'a2c_baseline_agent_history.csv'
AGENT_NAME = 'A2C Baseline Agent'

# How much to smooth the lines (e.g., 10 = average of every 10 episodes)
# This makes the graphs much easier to read
SMOOTHING_WINDOW = 10 
# ---------------------

def plot_agent_history(agent_name, csv_path):
    """Loads, processes, and plots the history for the baseline agent."""
    
    print(f"Processing {csv_path} for {agent_name}...")
    
    if not os.path.exists(csv_path):
        print(f"--- ERROR: File not found: {csv_path}. ---\n")
        return
        
    try:
        # Load the CSV data
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"--- WARNING: File is empty: {csv_path}. Skipping. ---\n")
        return
    except Exception as e:
        print(f"--- ERROR loading file {csv_path}: {e} ---\n")
        return

    if df.empty:
        print(f"--- WARNING: File is empty: {csv_path}. Skipping. ---\n")
        return

    # Calculate the moving average to smooth the graphs
    # 'min_periods=1' ensures it plots even if the window is large
    df['Reward_Smooth'] = df['Total_Reward'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    df['CLoss_Smooth'] = df['Critic_Loss'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    df['Entropy_Smooth'] = df['Entropy'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    
    # --- Create the plots for this agent ---
    
    # We create 3 subplots (vertical) for this one agent
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'A2C Training Metrics: {agent_name}', fontsize=18, y=1.02)

    # 1. Plot Episode Reward
    ax1.plot(df['Episode'], df['Total_Reward'], 'b-', alpha=0.2, label='Raw Reward')
    ax1.plot(df['Episode'], df['Reward_Smooth'], 'b-', linewidth=2, label=f'Smoothed Reward (Window={SMOOTHING_WINDOW})')
    ax1.set_ylabel('Total Episode Reward')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Episode Reward over Time')

    # 2. Plot Critic Loss
    ax2.plot(df['Episode'], df['Critic_Loss'], 'r-', alpha=0.2, label='Raw Loss')
    ax2.plot(df['Episode'], df['CLoss_Smooth'], 'r-', linewidth=2, label=f'Smoothed Loss (Window={SMOOTHING_WINDOW})')
    ax2.set_ylabel('Critic Loss (CLoss)')
    # Use log scale to see the spikes and the stable state
    # This will clearly show the CLoss exploding
    ax2.set_yscale('log') 
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Critic Loss over Time (Log Scale)')

    # 3. Plot Entropy
    ax3.plot(df['Episode'], df['Entropy'], 'g-', alpha=0.2, label='Raw Entropy')
    ax3.plot(df['Episode'], df['Entropy_Smooth'], 'g-', linewidth=2, label=f'Smoothed Entropy (Window={SMOOTHING_WINDOW})')
    ax3.set_xlabel('Episode Number')
    ax3.set_ylabel('Policy Entropy')
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Policy Entropy over Time')

    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(INPUT_DIR, f'{AGENT_NAME}_training_plot.png')
    plt.savefig(save_path, dpi=300)
    print(f"✓ Plot saved to {save_path}\n")
    plt.close(fig)

def main():
    print("="*70)
    print("Generating training plots from BASELINE CSV data...")
    print(f"Using smoothing window: {SMOOTHING_WINDOW}")
    print("="*70)
    
    csv_path = os.path.join(INPUT_DIR, CSV_FILE)
    plot_agent_history(AGENT_NAME, csv_path)
        
    print("All plots generated.")

if __name__ == "__main__":
    # Check for required libraries
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("--- ERROR ---")
        print("This script requires 'pandas' and 'matplotlib'.")
        print("Please install them by running:")
        print("pip install pandas matplotlib")
        sys.exit(1)
        
    main()