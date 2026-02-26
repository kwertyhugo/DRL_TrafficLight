import pandas as pd
import matplotlib.pyplot as plt
import os
import sys  

INPUT_DIR = r'Balibago_traci\output_DQN'

# Files to plot with labels
FILES_TO_PLOT = {
    'Baseline North': 'BP_North_history.csv',
    'Baseline South': 'BP_South_history.csv',
    'Signalized North': 'North_historyTEST.csv',
    'Signalized South': 'South_historyTEST.csv',
}

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Balibago DQN Training Results - All Agents', fontsize=16, fontweight='bold')

# Flatten axes for easier iteration
axes = axes.flatten()

for idx, (label, filename) in enumerate(FILES_TO_PLOT.items()):
    try:
        df = pd.read_csv(os.path.join(INPUT_DIR, filename))
        axes[idx].plot(df['Step'], df['Reward'], linewidth=1.5)
        axes[idx].set_title(label, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Training Step')
        axes[idx].set_ylabel('Reward')
        axes[idx].grid(True, alpha=0.3)
    except FileNotFoundError:
        axes[idx].text(0.5, 0.5, f'File not found:\n{filename}', 
                      ha='center', va='center', transform=axes[idx].transAxes,
                      fontsize=10, color='red')
        axes[idx].set_title(label, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('balibago_dqn_all_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("Plot saved as 'balibago_dqn_all_results.png'")
