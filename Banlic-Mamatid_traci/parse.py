import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os

def extract_timeloss_stats(filepath):
    """Extract mean and std dev of timeLoss for vehicles and pedestrians."""
    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        vehicle_time_losses = []
        pedestrian_time_losses = []

        for trip in root.findall('tripinfo'):
            tl = trip.get('timeLoss')
            if tl is not None:
                vehicle_time_losses.append(float(tl))

        for person in root.findall('personinfo'):
            for walk in person.findall('walk'):
                tl = walk.get('timeLoss')
                if tl is not None:
                    pedestrian_time_losses.append(float(tl))

        def stats(data):
            if not data:
                return None, None, 0
            return np.mean(data), np.std(data), len(data)

        v_mean, v_std, v_count = stats(vehicle_time_losses)
        p_mean, p_std, p_count = stats(pedestrian_time_losses)

        return {
            'vehicle_mean':  v_mean,
            'vehicle_std':   v_std,
            'vehicle_count': v_count,
            'pedestrian_mean':  p_mean,
            'pedestrian_std':   p_std,
            'pedestrian_count': p_count,
        }

    except Exception as e:
        print(f"  ERROR processing {filepath}: {e}")
        return {
            'vehicle_mean': None, 'vehicle_std': None, 'vehicle_count': 0,
            'pedestrian_mean': None, 'pedestrian_std': None, 'pedestrian_count': 0,
        }


# ── File definitions ──────────────────────────────────────────────────────────
BASE_DIR = r'Banlic-Mamatid_traci\output_A2C'

files = {
    'Normal':    os.path.join(BASE_DIR, 'test_normal_traffic_trips.xml'),
    'Slow':      os.path.join(BASE_DIR, 'test_slow_traffic_trips.xml'),
    'Jam/Heavy': os.path.join(BASE_DIR, 'test_jam_heavy_traffic_trips.xml'),
}

# ── Process ───────────────────────────────────────────────────────────────────
results = {}

print("=" * 65)
print("  Banlic-Mamatid  |  A2C  |  Trip Time-Loss Statistics")
print("=" * 65)

for scenario, filepath in files.items():
    print(f"\nScenario: {scenario}")
    if not os.path.exists(filepath):
        print(f"  ⚠  File not found: {filepath}")
        continue

    stats = extract_timeloss_stats(filepath)
    results[scenario] = stats

    print(f"  Vehicles   — count: {stats['vehicle_count']:>5}", end='')
    if stats['vehicle_mean'] is not None:
        print(f"  |  mean: {stats['vehicle_mean']:>8.2f} s  |  std: {stats['vehicle_std']:>8.2f} s")
    else:
        print("  |  no data")

    print(f"  Pedestrians— count: {stats['pedestrian_count']:>5}", end='')
    if stats['pedestrian_mean'] is not None:
        print(f"  |  mean: {stats['pedestrian_mean']:>8.2f} s  |  std: {stats['pedestrian_std']:>8.2f} s")
    else:
        print("  |  no data")

# ── Summary table ─────────────────────────────────────────────────────────────
rows = []
for scenario, s in results.items():
    rows.append({
        'Scenario':         scenario,
        'Veh_Count':        s['vehicle_count'],
        'Veh_Mean (s)':     round(s['vehicle_mean'],  2) if s['vehicle_mean']  is not None else None,
        'Veh_Std (s)':      round(s['vehicle_std'],   2) if s['vehicle_std']   is not None else None,
        'Ped_Count':        s['pedestrian_count'],
        'Ped_Mean (s)':     round(s['pedestrian_mean'],2) if s['pedestrian_mean'] is not None else None,
        'Ped_Std (s)':      round(s['pedestrian_std'], 2) if s['pedestrian_std']  is not None else None,
    })

df = pd.DataFrame(rows)

print("\n\n" + "=" * 65)
print("SUMMARY TABLE")
print("=" * 65)
print(df.to_string(index=False))

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_csv = 'banlic_mamatid_a2c_timeloss_stats.csv'
df.to_csv(out_csv, index=False)
print(f"\n✓ Results saved to '{out_csv}'")

# ── LaTeX table ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("LaTeX TABLE")
print("=" * 65)
print(r"""
\begin{table}[h]
\centering
\caption{A2C Time Loss Statistics -- Banlic-Mamatid Junction}
\label{tab:banlic_mamatid_a2c}
\begin{tabular}{lcccccc}
\hline
 & \multicolumn{3}{c}{Vehicles} & \multicolumn{3}{c}{Pedestrians} \\
Scenario & $n$ & Mean (s) & SD (s) & $n$ & Mean (s) & SD (s) \\
\hline""")

for _, row in df.iterrows():
    vm = f"{row['Veh_Mean (s)']:.2f}"  if row['Veh_Mean (s)'] is not None else '--'
    vs = f"{row['Veh_Std (s)']:.2f}"   if row['Veh_Std (s)']  is not None else '--'
    pm = f"{row['Ped_Mean (s)']:.2f}"  if row['Ped_Mean (s)'] is not None else '--'
    ps = f"{row['Ped_Std (s)']:.2f}"   if row['Ped_Std (s)']  is not None else '--'
    print(f"{row['Scenario']:<12} & {int(row['Veh_Count'])} & {vm} & {vs} "
          f"& {int(row['Ped_Count'])} & {pm} & {ps} \\\\")

print(r"""\hline
\end{tabular}
\end{table}""")