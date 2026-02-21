import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
INPUT_DIR = r'Olivarez_traci\output_DQN'

FILES_TO_PLOT = {
    'Main Agent': 'main_agent_history.csv',
}

SMOOTHING_WINDOW = 50
SKIP_ROWS = 29836
EPSILON_HEAD_ROWS = 1500
# ---------------------

def plot_agent_history(agent_name, csv_path):
    """Loads, processes, and plots the DQN history for a single agent."""
    print(f"Processing {csv_path} for {agent_name}...")

    if not os.path.exists(csv_path):
        print(f"--- WARNING: File not found: {csv_path}. Skipping. ---\n")
        return

    try:
        df     = pd.read_csv(csv_path, skiprows=range(1, SKIP_ROWS + 1))
        df_eps = pd.read_csv(csv_path, nrows=EPSILON_HEAD_ROWS)
    except pd.errors.EmptyDataError:
        print(f"--- WARNING: File is empty: {csv_path}. Skipping. ---\n")
        return
    except Exception as e:
        print(f"--- ERROR loading file {csv_path}: {e} ---\n")
        return

    if df.empty:
        print(f"--- WARNING: DataFrame is empty after loading: {csv_path}. Skipping. ---\n")
        return

    expected_cols = {'Step', 'Reward', 'Loss', 'Epsilon'}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"--- ERROR: Missing columns {missing} in {csv_path}. Skipping. ---\n")
        return

    print(f"  Skipped first {SKIP_ROWS:,} rows. Plotting {len(df):,} remaining rows.")

    df = df.reset_index(drop=True)
    x  = df.index

    MAX_RAW_POINTS = 10_000
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

    df_eps = df_eps.reset_index(drop=True)
    x_eps  = df_eps.index

    if len(df_eps) > MAX_RAW_POINTS:
        step_e      = len(df_eps) // MAX_RAW_POINTS
        x_eps_raw   = x_eps[::step_e]
        epsilon_raw = df_eps['Epsilon'].iloc[::step_e]
    else:
        x_eps_raw   = x_eps
        epsilon_raw = df_eps['Epsilon']

    epsilon_smooth = df_eps['Epsilon'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle(f'DQN Training Metrics: {agent_name}', fontsize=18, y=1.02)

    ax1.plot(x_raw, reward_raw,    'b-', alpha=0.15, linewidth=0.5, label='Raw Reward')
    ax1.plot(x,     reward_smooth, 'b-', linewidth=2,
             label=f'Smoothed Reward (Window={SMOOTHING_WINDOW})')
    ax1.set_ylabel('Total Step Reward')
    ax1.set_xlabel(f'Row Index (starting from row {SKIP_ROWS:,})')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title(f'Reward over Time  (first {SKIP_ROWS:,} rows skipped)')

    ax2.plot(x_raw, loss_raw,    'r-', alpha=0.15, linewidth=0.5, label='Raw Loss')
    ax2.plot(x,     loss_smooth, 'r-', linewidth=2,
             label=f'Smoothed Loss (Window={SMOOTHING_WINDOW})')
    ax2.set_ylabel('TD Loss')
    ax2.set_xlabel(f'Row Index (starting from row {SKIP_ROWS:,})')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('TD Loss over Time (Log Scale)')

    ax3.plot(x_eps_raw, epsilon_raw,    'g-', alpha=0.15, linewidth=0.5, label='Raw Epsilon')
    ax3.plot(x_eps,     epsilon_smooth, 'g-', linewidth=2,
             label=f'Smoothed Epsilon (Window={SMOOTHING_WINDOW})')
    ax3.set_xlabel('')
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.set_ylabel('Epsilon (Exploration Rate)')
    ax3.set_ylim(0, 1.05)
    ax3.legend()
    ax3.grid(True)
    ax3.set_title('Epsilon Decay over Time')

    plt.tight_layout()
    save_path = os.path.join(INPUT_DIR, f'{agent_name}_dqn_training_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path}\n")
    plt.close(fig)


def main():
    print("=" * 70)
    print("Generating DQN training plots from CSV data...")
    print(f"Using smoothing window : {SMOOTHING_WINDOW}")
    print(f"Skipping first rows    : {SKIP_ROWS:,}")
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