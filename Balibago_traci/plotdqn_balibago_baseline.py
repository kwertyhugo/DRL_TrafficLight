import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
INPUT_DIR = r'Balibago_traci\output_DQN'

FILES_TO_PLOT = {
    'Baseline North': 'BP_North_history.csv',
    'Balibago Baseline Agent': 'BP_South_history.csv',
}

SMOOTHING_WINDOW = 15
EPSILON_HEAD_ROWS = 500
# ---------------------

def plot_agent_history(agent_name, csv_path):
    """Loads, processes, and plots the DQN history for a single baseline agent."""
    print(f"Processing {csv_path} for {agent_name}...")

    if not os.path.exists(csv_path):
        print(f"--- WARNING: File not found: {csv_path}. Skipping. ---\n")
        return

    try:
        df_full = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        print(f"--- WARNING: File is empty: {csv_path}. Skipping. ---\n")
        return
    except Exception as e:
        print(f"--- ERROR loading file {csv_path}: {e} ---\n")
        return

    if df_full.empty:
        print(f"--- WARNING: DataFrame is empty after loading: {csv_path}. Skipping. ---\n")
        return

    expected_cols = {'Step', 'Reward', 'Loss', 'Epsilon'}
    missing = expected_cols - set(df_full.columns)
    if missing:
        print(f"--- ERROR: Missing columns {missing} in {csv_path}. Skipping. ---\n")
        return

    total_rows = len(df_full)
    print(f"  Total rows: {total_rows:,}. Plotting all rows.")

    df = df_full.reset_index(drop=True)
    x  = df.index

    MAX_RAW_POINTS = 2_000
    if len(df) > MAX_RAW_POINTS:
        step       = len(df) // MAX_RAW_POINTS
        x_raw      = x[::step]
        reward_raw = df['Reward'].iloc[::step]
        loss_raw   = df['Loss'].iloc[::step]
    else:
        x_raw      = x
        reward_raw = df['Reward']
        loss_raw   = df['Loss']

    reward_smooth = df['Reward'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
    loss_smooth   = df['Loss'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # For epsilon, use the first portion of the data
    epsilon_rows = min(EPSILON_HEAD_ROWS, len(df_full))
    df_eps = df_full.iloc[:epsilon_rows].reset_index(drop=True)
    x_eps  = df_eps.index

    if len(df_eps) > MAX_RAW_POINTS:
        step_e      = len(df_eps) // MAX_RAW_POINTS
        x_eps_raw   = x_eps[::step_e]
        epsilon_raw = df_eps['Epsilon'].iloc[::step_e]
    else:
        x_eps_raw   = x_eps
        epsilon_raw = df_eps['Epsilon']

    epsilon_smooth = df_eps['Epsilon'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    # Plot 1: Reward
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(x_raw, reward_raw,    'b-', alpha=0.15, linewidth=0.5, label='Raw Reward')
    ax1.plot(x,     reward_smooth, 'b-', linewidth=2,
             label=f'Smoothed Reward (Window={SMOOTHING_WINDOW})')
    ax1.set_ylabel('Total Step Reward', fontsize=12)
    ax1.set_xlabel('Row Index', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True)
    ax1.set_title(f'DQN {agent_name}: Reward over Time', fontsize=14)
    plt.tight_layout()
    save_path1 = os.path.join(INPUT_DIR, f'{agent_name}_reward.png')
    plt.savefig(save_path1, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path1}")
    plt.close(fig1)

    # Plot 2: Loss
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(x_raw, loss_raw,    'r-', alpha=0.15, linewidth=0.5, label='Raw Loss')
    ax2.plot(x,     loss_smooth, 'r-', linewidth=2,
             label=f'Smoothed Loss (Window={SMOOTHING_WINDOW})')
    ax2.set_ylabel('TD Loss', fontsize=12)
    ax2.set_xlabel('Row Index', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True)
    ax2.set_title(f'DQN {agent_name}: TD Loss over Time (Log Scale)', fontsize=14)
    plt.tight_layout()
    save_path2 = os.path.join(INPUT_DIR, f'{agent_name}_loss.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path2}")
    plt.close(fig2)

    # Plot 3: Epsilon
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(x_eps_raw, epsilon_raw,    'g-', alpha=0.15, linewidth=0.5, label='Raw Epsilon')
    ax3.plot(x_eps,     epsilon_smooth, 'g-', linewidth=2,
             label=f'Smoothed Epsilon (Window={SMOOTHING_WINDOW})')
    ax3.set_xlabel('Row Index', fontsize=12)
    ax3.set_ylabel('Epsilon (Exploration Rate)', fontsize=12)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=11)
    ax3.grid(True)
    ax3.set_title(f'DQN {agent_name}: Epsilon Decay over Time', fontsize=14)
    plt.tight_layout()
    save_path3 = os.path.join(INPUT_DIR, f'{agent_name}_epsilon.png')
    plt.savefig(save_path3, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path3}\n")
    plt.close(fig3)


def main():
    print("=" * 70)
    print("Generating DQN training plots for Balibago (Baseline)...")
    print(f"Using smoothing window : {SMOOTHING_WINDOW}")
    print("=" * 70)

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
