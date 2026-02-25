import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import shutil

# --- Configuration ---
INPUT_DIR = r'Balibago_traci\output_A2C'
OUTPUT_DIR = r'Plots\Balibago A2C'

FILES_TO_PLOT = {
    'Agent': 'South_A2C_history.csv',
}

SMOOTHING_WINDOW = 10
# ---------------------

def plot_agent_history(agent_name, csv_path):
    """Loads, processes, and plots the A2C history for a single agent."""
    print(f"Processing {csv_path} for {agent_name}...")

    if not os.path.exists(csv_path):
        print(f"--- WARNING: File not found: {csv_path}. Skipping. ---\n")
        return

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"--- WARNING: File is empty: {csv_path}. Skipping. ---\n")
        return
    except Exception as e:
        print(f"--- ERROR loading file {csv_path}: {e} ---\n")
        return

    if df.empty:
        print(f"--- WARNING: DataFrame is empty after loading: {csv_path}. Skipping. ---\n")
        return

    expected_cols = {'Episode', 'Total_Reward', 'Critic_Loss', 'Entropy'}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"--- ERROR: Missing columns {missing} in {csv_path}. Skipping. ---\n")
        return

    total_rows = len(df)
    print(f"  Total episodes: {total_rows:,}. Plotting all episodes.")

    df = df.reset_index(drop=True)
    x = df.index

    MAX_RAW_POINTS = 2_000
    if len(df) > MAX_RAW_POINTS:
        step       = len(df) // MAX_RAW_POINTS
        x_raw      = x[::step]
        reward_raw = df['Total_Reward'].iloc[::step]
        loss_raw   = df['Critic_Loss'].iloc[::step]
    else:
        x_raw      = x
        reward_raw = df['Total_Reward']
        loss_raw   = df['Critic_Loss']

    reward_smooth = df['Total_Reward'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    loss_smooth   = df['Critic_Loss'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    entropy_smooth = df['Entropy'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Plot 1: Reward
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x_raw, reward_raw,    'b-', alpha=0.15, linewidth=0.5, label='Raw Reward')
    ax1.plot(x,     reward_smooth, 'b-', linewidth=2,
             label=f'Smoothed Reward (Window={SMOOTHING_WINDOW})')
    ax1.set_ylabel('Total Episode Reward', fontsize=12)
    ax1.set_xlabel('Episode Number', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True)
    ax1.set_title(f'A2C {agent_name}: Reward over Time', fontsize=14)
    plt.tight_layout()
    save_path1 = os.path.join(OUTPUT_DIR, f'{agent_name}_reward.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path1}")
    plt.close(fig1)

    # Plot 2: Critic Loss
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(x_raw, loss_raw,    'r-', alpha=0.15, linewidth=0.5, label='Raw Loss')
    ax2.plot(x,     loss_smooth, 'r-', linewidth=2,
             label=f'Smoothed Loss (Window={SMOOTHING_WINDOW})')
    ax2.set_ylabel('Critic Loss', fontsize=12)
    ax2.set_xlabel('Episode Number', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True)
    ax2.set_title(f'A2C {agent_name}: Critic Loss over Time (Log Scale)', fontsize=14)
    plt.tight_layout()
    save_path2 = os.path.join(OUTPUT_DIR, f'{agent_name}_loss.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path2}")
    plt.close(fig2)

    # Plot 3: Entropy
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(x, df['Entropy'],    'g-', alpha=0.15, linewidth=0.5, label='Raw Entropy')
    ax3.plot(x, entropy_smooth, 'g-', linewidth=2,
             label=f'Smoothed Entropy (Window={SMOOTHING_WINDOW})')
    ax3.set_xlabel('Episode Number', fontsize=12)
    ax3.set_ylabel('Policy Entropy', fontsize=12)
    ax3.legend(fontsize=11)
    ax3.grid(True)
    ax3.set_title(f'A2C {agent_name}: Policy Entropy over Time', fontsize=14)
    plt.tight_layout()
    save_path3 = os.path.join(OUTPUT_DIR, f'{agent_name}_entropy.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path3}\n")
    plt.close(fig3)


def main():
    print("=" * 70)
    print("Generating A2C training plots for Balibago (Signalized)...")
    print(f"Using smoothing window : {SMOOTHING_WINDOW}")
    print(f"Output directory       : {OUTPUT_DIR}")
    print("=" * 70)

    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for agent_name, file_name in FILES_TO_PLOT.items():
        csv_path = os.path.join(INPUT_DIR, file_name)
        plot_agent_history(agent_name, csv_path)

    print("All plots generated.")


if __name__ == "__main__":
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        print("--- ERROR ---")
        print("This script requires 'pandas' and 'matplotlib'.")
        print("Please install them by running:")
        print("  pip install pandas matplotlib")
        sys.exit(1)

    main()
