import os
import sys

# --- Configuration ---
INPUT_DIR = r'Balibago_traci\Plots\BALIBAGO DQN'
PREFIX = '[DQN]'
# ---------------------

def rename_files():
    """Renames all files in the directory to add [DQN] prefix."""
    
    if not os.path.exists(INPUT_DIR):
        print(f"--- ERROR: Directory not found: {INPUT_DIR} ---\n")
        return False
    
    try:
        files = os.listdir(INPUT_DIR)
    except Exception as e:
        print(f"--- ERROR reading directory {INPUT_DIR}: {e} ---\n")
        return False
    
    if not files:
        print(f"--- WARNING: No files found in {INPUT_DIR} ---\n")
        return True
    
    renamed_count = 0
    skipped_count = 0
    
    print("=" * 70)
    print(f"Renaming files in {INPUT_DIR}")
    print(f"Adding prefix: {PREFIX}")
    print("=" * 70)
    
    for file_name in files:
        file_path = os.path.join(INPUT_DIR, file_name)
        
        # Skip if not a file (e.g., directories)
        if not os.path.isfile(file_path):
            continue
        
        # Skip if already has the prefix
        if file_name.startswith(PREFIX):
            print(f"⊘ Skipped (already has prefix): {file_name}")
            skipped_count += 1
            continue
        
        # Create new name
        new_name = f"{PREFIX} {file_name}"
        new_path = os.path.join(INPUT_DIR, new_name)
        
        try:
            os.rename(file_path, new_path)
            print(f"✓ Renamed: {file_name} → {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"✗ ERROR renaming {file_name}: {e}")
    
    print("=" * 70)
    print(f"Renamed: {renamed_count} files")
    print(f"Skipped: {skipped_count} files")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = rename_files()
    sys.exit(0 if success else 1)
