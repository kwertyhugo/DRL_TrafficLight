"""
run_flow_simulation.py

Script to run SUMO simulation with flow-based traffic generation.
This handles the complete flow-based simulation process.
"""

import subprocess
import sys
from pathlib import Path
import time

# Configuration
SCRIPT_DIR = Path(__file__).parent
SUMO_CONFIG = SCRIPT_DIR / 'flows.sumocfg'
FLOWS_FILE = SCRIPT_DIR / 'demand' / 'flows.flows.xml'
OUTPUT_DIR = SCRIPT_DIR / 'output'

def check_prerequisites():
    """Check if all required files exist"""
    print("🔍 CHECKING PREREQUISITES")
    print("=" * 50)
    
    missing_files = []
    
    # Check for flows file
    if not FLOWS_FILE.exists():
        missing_files.append(f"Flows file: {FLOWS_FILE}")
        print(f"❌ Missing: {FLOWS_FILE}")
        print("   Run: python generate_flows_from_taz.py")
    else:
        print(f"✅ Found flows file: {FLOWS_FILE}")
    
    # Check for SUMO config
    if not SUMO_CONFIG.exists():
        missing_files.append(f"SUMO config: {SUMO_CONFIG}")
        print(f"❌ Missing: {SUMO_CONFIG}")
    else:
        print(f"✅ Found SUMO config: {SUMO_CONFIG}")
    
    # Check for network file
    network_file = SCRIPT_DIR / 'network' / 'map.net.xml'
    if not network_file.exists():
        missing_files.append(f"Network file: {network_file}")
        print(f"❌ Missing: {network_file}")
    else:
        print(f"✅ Found network file: {network_file}")
    
    # Check for TAZ file
    taz_file = SCRIPT_DIR / 'network' / 'traffic.taz.xml'
    if not taz_file.exists():
        missing_files.append(f"TAZ file: {taz_file}")
        print(f"❌ Missing: {taz_file}")
    else:
        print(f"✅ Found TAZ file: {taz_file}")
    
    return len(missing_files) == 0, missing_files

def run_simulation(gui=False):
    """Run SUMO simulation with flows"""
    print(f"\n🚗 RUNNING SUMO SIMULATION WITH FLOWS")
    print("=" * 50)
    
    # Choose SUMO executable
    sumo_cmd = 'sumo-gui' if gui else 'sumo'
    
    cmd = [
        sumo_cmd,
        '-c', str(SUMO_CONFIG),
        '--start', 'true' if gui else 'false',
        '--quit-on-end', 'true' if not gui else 'false'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Config file: {SUMO_CONFIG}")
    print(f"Using {'GUI' if gui else 'command-line'} mode")
    
    if gui:
        print("🖥️  SUMO GUI will open - you can start/stop/pause the simulation")
        print("   Click the 'Play' button to start the simulation")
    else:
        print("⏳ Running simulation in background...")
    
    try:
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            capture_output=not gui,  # Capture output only in non-GUI mode
            text=True
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Simulation completed successfully!")
            print(f"⏱️  Duration: {duration:.1f} seconds")
            
            # Check output files
            print(f"\n📊 Output files generated:")
            output_files = list(OUTPUT_DIR.glob("*.xml"))
            for output_file in output_files:
                size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"   - {output_file.name} ({size_mb:.1f} MB)")
            
            if not gui and result.stdout:
                print(f"\n📝 Simulation summary:")
                # Extract key statistics from SUMO output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'vehicles' in line.lower() or 'simulation' in line.lower():
                        print(f"   {line.strip()}")
            
        else:
            print(f"❌ Simulation failed with return code {result.returncode}")
            if not gui:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
            return False
            
    except FileNotFoundError:
        print(f"❌ {sumo_cmd} not found. Please ensure SUMO is installed and in PATH.")
        print("   Install SUMO: https://eclipse.dev/sumo/")
        return False
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False
    
    return True

def main():
    print("🌊 SUMO FLOW-BASED SIMULATION RUNNER")
    print("=" * 50)
    
    # Check prerequisites
    ready, missing = check_prerequisites()
    
    if not ready:
        print(f"\n❌ Cannot run simulation. Missing files:")
        for missing_file in missing:
            print(f"   - {missing_file}")
        print(f"\n💡 To fix:")
        print(f"   1. Run: python generate_flows_from_taz.py")
        print(f"   2. Ensure all network files exist")
        sys.exit(1)
    
    # Ask user preference for GUI vs command-line
    print(f"\n🎮 How would you like to run the simulation?")
    print(f"   1. GUI mode (visual, interactive)")
    print(f"   2. Command-line mode (faster, background)")
    
    choice = input("Enter choice (1 or 2, default=2): ").strip()
    use_gui = choice == '1'
    
    # Run simulation
    success = run_simulation(gui=use_gui)
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print("=" * 50)
        print(f"✅ Flow-based simulation completed")
        print(f"📁 Output files saved in: {OUTPUT_DIR}")
        print(f"🔍 Analyze results using SUMO tools or custom scripts")
        
        if not use_gui:
            print(f"\n💡 To visualize results:")
            print(f"   sumo-gui -c {SUMO_CONFIG}")
    else:
        print(f"\n❌ Simulation failed. Check error messages above.")

if __name__ == '__main__':
    main()