import subprocess
import sys

# --- Configuration ---
PLOT_SCRIPTS = [
    ('A2C Signalized', r'Balibago_traci\plotA2C_balibago.py'),
    ('A2C Baseline', r'Balibago_traci\plotA2C_balibago_baseline.py'),
    ('DDPG Signalized', r'Balibago_traci\plotDDPG_balibago.py'),
    ('DDPG Baseline', r'Balibago_traci\plotDDPG_balibago_baseline.py'),
]

RENAME_SCRIPTS = [
    ('A2C', r'rename_a2c_plots.py'),
    ('DDPG', r'rename_ddpg_plots.py'),
]
# ---------------------

def run_script(script_name, script_path):
    """Run a Python script and report results."""
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run([sys.executable, script_path], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR running {script_name}: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        return False


def main():
    print("="*70)
    print("BALIBAGO DRL - A2C & DDPG PLOT GENERATION & RENAMING")
    print("="*70)
    
    print("\nStep 1: Generating plots...")
    plot_results = []
    for name, script in PLOT_SCRIPTS:
        success = run_script(name, script)
        plot_results.append((name, success))
    
    print("\nStep 2: Renaming plot files...")
    rename_results = []
    for name, script in RENAME_SCRIPTS:
        success = run_script(name, script)
        rename_results.append((name, success))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nPlot Generation Results:")
    for name, success in plot_results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    print("\nFile Renaming Results:")
    for name, success in rename_results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {name}")
    
    all_success = all(s for _, s in plot_results + rename_results)
    
    print("\n" + "="*70)
    if all_success:
        print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
    else:
        print("⚠ SOME TASKS FAILED - PLEASE CHECK ABOVE")
    print("="*70)
    
    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
