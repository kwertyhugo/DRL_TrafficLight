import xml.etree.ElementTree as ET
import numpy as np
import os
from pathlib import Path

# --- Configuration ---
INPUT_DIR = r'Balibago_traci\output_DQN'

# Files to analyze (map test name to file path)
TEST_FILES = {
    'BP Jam Traffic': 'BP_DQN_trips_JAM.xml',
    'BP Normal Traffic': 'BP_DQN_trips_NORMALTEST2.xml',
    'BP Slow Traffic': 'BP_DQN_trips_SLOW.xml',
    'SD Slow Traffic': 'SD_DQN_Train_Tripinfo_SLOW2.xml',
    'SD Normal Traffic': 'SD_DQN_Train_Tripinfo_NORMAL2.xml',
    'SD Jam Traffic': 'SD_DQN_Train_Tripinfo_JAM.xml',
}

# Metrics to extract from tripinfo (vehicle trips)
METRICS_TO_EXTRACT = {
    'duration': 'Trip Duration (s)',
    'routeLength': 'Route Length (m)',
    'waitingTime': 'Waiting Time (s)',
    'waitingCount': 'Waiting Count',
    'timeLoss': 'Time Loss (s)',
}
# ---------------------

def parse_xml_and_extract_metrics(file_path):
    """Parse XML file and extract metrics from tripinfo elements."""
    
    if not os.path.exists(file_path):
        print(f"  ✗ File not found: {file_path}")
        return None
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"  ✗ ERROR parsing XML: {e}")
        return None
    
    # Extract tripinfo elements (vehicles) - exclude personinfo
    tripinfos = root.findall('tripinfo')
    
    if not tripinfos:
        print(f"  ✗ No tripinfo elements found in {file_path}")
        return None
    
    metrics_data = {key: [] for key in METRICS_TO_EXTRACT.keys()}
    
    for tripinfo in tripinfos:
        for metric_key in METRICS_TO_EXTRACT.keys():
            try:
                value = float(tripinfo.get(metric_key, 0))
                metrics_data[metric_key].append(value)
            except (ValueError, TypeError):
                pass
    
    return metrics_data


def calculate_statistics(metrics_data):
    """Calculate mean and standard deviation for each metric."""
    
    if not metrics_data:
        return None
    
    statistics = {}
    for metric_key, values in metrics_data.items():
        if values:
            statistics[metric_key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values),
            }
    
    return statistics


def print_results(test_name, file_path, statistics):
    """Print formatted results for a test."""
    
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"File: {file_path}")
    print(f"{'='*80}")
    
    if not statistics:
        print("  No data available\n")
        return
    
    for metric_key, metric_label in METRICS_TO_EXTRACT.items():
        if metric_key in statistics:
            stats = statistics[metric_key]
            print(f"\n{metric_label}:")
            print(f"  Count  : {stats['count']:.0f} vehicles")
            print(f"  Mean   : {stats['mean']:.2f}")
            print(f"  StdDev : {stats['std']:.2f}")
            print(f"  Min    : {stats['min']:.2f}")
            print(f"  Max    : {stats['max']:.2f}")


def main():
    print("="*80)
    print("DQN TEST METRICS ANALYSIS - BALIBAGO")
    print("="*80)
    print("\nAnalyzing vehicle trip metrics from XML files...\n")
    
    all_results = {}
    
    for test_name, file_name in TEST_FILES.items():
        file_path = os.path.join(INPUT_DIR, file_name)
        print(f"Processing: {test_name}... ", end='', flush=True)
        
        # Parse and extract metrics
        metrics_data = parse_xml_and_extract_metrics(file_path)
        
        if metrics_data:
            print("✓")
            # Calculate statistics
            statistics = calculate_statistics(metrics_data)
            all_results[test_name] = (file_path, statistics)
        else:
            print("✗")
    
    # Print detailed results for each test
    for test_name, (file_path, statistics) in all_results.items():
        print_results(test_name, file_path, statistics)
    
    # Print summary comparison table
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    if all_results:
        # Get all metric keys
        all_metrics = set()
        for _, statistics in all_results.values():
            if statistics:
                all_metrics.update(statistics.keys())
        
        for metric_key in sorted(all_metrics):
            metric_label = METRICS_TO_EXTRACT.get(metric_key, metric_key)
            print(f"\n{metric_label}:")
            print(f"  {'Test':<20} {'Mean':<12} {'StdDev':<12} {'Vehicles':<10}")
            print(f"  {'-'*54}")
            
            for test_name, (_, statistics) in all_results.items():
                if statistics and metric_key in statistics:
                    stats = statistics[metric_key]
                    print(f"  {test_name:<20} {stats['mean']:<12.2f} {stats['std']:<12.2f} {stats['count']:<10.0f}")
    
    print(f"\n{'='*80}")
    print("Analysis complete.")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
