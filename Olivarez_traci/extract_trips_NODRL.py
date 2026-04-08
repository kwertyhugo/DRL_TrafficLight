import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os

def extract_timeloss_stats(tripinfo_file):
    """Extract mean and std dev of timeLoss from SUMO tripinfo file"""
    try:
        tree = ET.parse(tripinfo_file)
        root = tree.getroot()
        
        time_losses = [float(trip.get('timeLoss')) 
                       for trip in root.findall('tripinfo')]
        
        if len(time_losses) == 0:
            return {'mean': None, 'std': None, 'variance': None, 'count': 0}
        
        return {
            'mean': np.mean(time_losses),
            'std': np.std(time_losses),
            'variance': np.var(time_losses),
            'count': len(time_losses)
        }
    except Exception as e:
        print(f"Error processing {tripinfo_file}: {e}")
        return {'mean': None, 'std': None, 'variance': None, 'count': 0}

# Define scenarios
scenarios = {
    'Normal': 'normaltraffic',
    'Slow': 'slowtraffic', 
    'Jam': 'trafficjam'
}

# Process all files
all_results = {}

# No DRL Baseline (BP = Baseline Pedestrian)
print("=" * 60)
print("Processing No DRL - Baseline Pedestrian Crossing...")
print("=" * 60)
for scenario_name, scenario_file in scenarios.items():
    filepath = f'Olivarez_traci\\output_NoDRL\\BP_NoDRL_trips_{scenario_file}.xml'
    
    if os.path.exists(filepath):
        key = f'NoDRL_Baseline_{scenario_name}'
        print(f"Processing {key}...")
        stats = extract_timeloss_stats(filepath)
        all_results[key] = stats
        
        if stats['count'] > 0:
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.2f} seconds")
            print(f"  Std Dev: {stats['std']:.2f} seconds")
        print()
    else:
        print(f"File not found: {filepath}\n")

# No DRL Signalized (SP = Signalized Pedestrian)
print("=" * 60)
print("Processing No DRL - Signalized Pedestrian Crossing...")
print("=" * 60)
for scenario_name, scenario_file in scenarios.items():
    filepath = f'Olivarez_traci\\output_NoDRL\\SP_NoDRL_trips_{scenario_file}.xml'
    
    if os.path.exists(filepath):
        key = f'NoDRL_Signalized_{scenario_name}'
        print(f"Processing {key}...")
        stats = extract_timeloss_stats(filepath)
        all_results[key] = stats
        
        if stats['count'] > 0:
            print(f"  Count: {stats['count']}")
            print(f"  Mean: {stats['mean']:.2f} seconds")
            print(f"  Std Dev: {stats['std']:.2f} seconds")
        print()
    else:
        print(f"File not found: {filepath}\n")

# Create DataFrame
df = pd.DataFrame(all_results).T

# Reorganize for better viewing
df_reset = df.reset_index()
df_reset[['Algorithm', 'Crossing_Type', 'Scenario']] = df_reset['index'].str.split('_', n=2, expand=True)
df_reset = df_reset[['Algorithm', 'Crossing_Type', 'Scenario', 'mean', 'std', 'variance', 'count']]

print("\n" + "=" * 60)
print("COMPLETE RESULTS")
print("=" * 60)
print(df_reset.to_string(index=False))

# Save to CSV
df_reset.to_csv('olivarez_nodrl_results.csv', index=False)
print("\n✓ Results saved to 'olivarez_nodrl_results.csv'")

# Create comparison tables by scenario
print("\n" + "=" * 60)
print("COMPARISON BY SCENARIO")
print("=" * 60)

for scenario in scenarios.keys():
    print(f"\n{scenario.upper()} TRAFFIC:")
    print("-" * 50)
    scenario_data = df_reset[df_reset['Scenario'] == scenario]
    if not scenario_data.empty:
        comparison = scenario_data[['Crossing_Type', 'mean', 'std', 'count']]
        print(comparison.to_string(index=False))
    else:
        print("  No data available")

# Create side-by-side comparison table
print("\n" + "=" * 60)
print("SIDE-BY-SIDE COMPARISON TABLE")
print("=" * 60)

comparison_table = []
for scenario in scenarios.keys():
    baseline_data = df_reset[(df_reset['Crossing_Type'] == 'Baseline') & 
                              (df_reset['Scenario'] == scenario)]
    signalized_data = df_reset[(df_reset['Crossing_Type'] == 'Signalized') & 
                                (df_reset['Scenario'] == scenario)]
    
    row = {
        'Scenario': scenario,
        'Baseline_Mean': baseline_data['mean'].values[0] if len(baseline_data) > 0 else None,
        'Baseline_Std': baseline_data['std'].values[0] if len(baseline_data) > 0 else None,
        'Signalized_Mean': signalized_data['mean'].values[0] if len(signalized_data) > 0 else None,
        'Signalized_Std': signalized_data['std'].values[0] if len(signalized_data) > 0 else None,
    }
    
    # Calculate difference (positive = signalized is worse)
    if row['Baseline_Mean'] is not None and row['Signalized_Mean'] is not None:
        row['Difference'] = row['Signalized_Mean'] - row['Baseline_Mean']
        row['Change_%'] = (row['Difference'] / row['Baseline_Mean'] * 100)
    else:
        row['Difference'] = None
        row['Change_%'] = None
    
    comparison_table.append(row)

comparison_df = pd.DataFrame(comparison_table)
print(comparison_df.to_string(index=False))

# Save comparison table
comparison_df.to_csv('olivarez_nodrl_comparison.csv', index=False)
print("\n✓ Comparison table saved to 'olivarez_nodrl_comparison.csv'")

# Create LaTeX-ready table
print("\n" + "=" * 60)
print("LATEX TABLE FORMAT")
print("=" * 60)
print("\n% Copy this into your LaTeX document:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{No DRL Performance: Baseline vs Signalized Pedestrian Crossing - Olivarez Junction}")
print("\\label{tab:olivarez_nodrl}")
print("\\begin{tabular}{lcccc}")
print("\\hline")
print("Scenario & \\multicolumn{2}{c}{Baseline} & \\multicolumn{2}{c}{Signalized} \\\\")
print("         & Mean (s) & Std Dev (s) & Mean (s) & Std Dev (s) \\\\")
print("\\hline")

for _, row in comparison_df.iterrows():
    if row['Baseline_Mean'] is not None and row['Signalized_Mean'] is not None:
        print(f"{row['Scenario']:<12} & "
              f"{row['Baseline_Mean']:>6.2f} & "
              f"{row['Baseline_Std']:>6.2f} & "
              f"{row['Signalized_Mean']:>6.2f} & "
              f"{row['Signalized_Std']:>6.2f} \\\\")

print("\\hline")
print("\\end{tabular}")
print("\\end{table}")