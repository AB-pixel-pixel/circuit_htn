import time
import subprocess
import os
import glob
import sys

def get_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def run_step(name, command):
    print(f"\n[{name}] Starting...")
    print(f"Command: {command}")
    
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    duration = end_time - start_time
    
    if result.returncode != 0:
        print(f"[{name}] FAILED after {duration:.2f}s")
        print("Error Output:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"[{name}] Completed in {duration:.2f}s")
        # Print first few lines of stdout to verify
        print("Output head:")
        print("\n".join(result.stdout.splitlines()[:10]))
        # Check for specific success messages
        if "Found" in result.stdout:
             for line in result.stdout.splitlines():
                 if "Found" in line and "trajectories" in line:
                     print(f"Verified: {line}")
        return duration

def main():
    total_start = time.time()
    
    # 1. Extraction (Training)
    # This generates domain_knowledge/alfred_htn_XXX.pkl
    # We use '10' for testing speed, but user asked for "whole training set".
    # User said: "from entire training set trajectory training".
    # BEWARE: Processing 6000+ trajectories might take HOURS. 
    # However, I must follow instructions.
    # But for this environment, I'll check if I should run ALL. 
    # The previous `verify_htn` ran on the pre-generated one.
    # I will run it without arguments to process ALL.
    
    # Clean up old pkls to ensure we pick the new one? No, versioning handles it.
    
    time_extraction = run_step("1. Extraction (Construct HTN)", "python scripts/htn/construct_alfred_htn.py")
    
    # Find the generated file
    input_pkl = get_latest_file("domain_knowledge/alfred_htn_*.pkl")
    if not input_pkl or "lifted" in input_pkl or "refined" in input_pkl:
        # Filter out lifted/refined if glob caught them (unlikely with pattern, but be safe)
        # alfred_htn_005.pkl vs alfred_htn_lifted_005.pkl
        # My glob "alfred_htn_*.pkl" catches "alfred_htn_lifted..." too?
        # Yes.
        candidates = glob.glob("domain_knowledge/alfred_htn_*.pkl")
        candidates = [f for f in candidates if "lifted" not in f and "refined" not in f]
        input_pkl = max(candidates, key=os.path.getctime)
        
    print(f"-> Generated: {input_pkl}")
    
    # 2. Lifting
    # Output: domain_knowledge/alfred_htn_lifted_XXX.pkl
    time_lifting = run_step("2. Lifting", f"python scripts/htn/lift_alfred_htn.py {input_pkl}")
    
    lifted_pkl = input_pkl.replace("alfred_htn_", "alfred_htn_lifted_")
    if not os.path.exists(lifted_pkl):
        print(f"Error: Expected output {lifted_pkl} not found.")
        sys.exit(1)
        
    # 3. Refinement
    # Output: domain_knowledge/alfred_htn_refined_XXX.pkl
    time_refinement = run_step("3. Refinement", f"python scripts/htn/refine_lifted_htn.py {lifted_pkl}")
    
    refined_pkl = input_pkl.replace("alfred_htn_", "alfred_htn_refined_")
    
    # Optional: Convert to Py (usually fast, maybe not counted in "Refine" strict sense but useful)
    # run_step("4. Code Gen", f"python scripts/htn/convert_pkl_to_py.py {refined_pkl}")

    total_end = time.time()
    total_duration = total_end - total_start
    
    print("\n" + "="*40)
    print("       TIMING REPORT       ")
    print("="*40)
    print(f"1. Extraction : {time_extraction:8.2f} s")
    print(f"2. Lifting    : {time_lifting:8.2f} s")
    print(f"3. Refinement : {time_refinement:8.2f} s")
    print("-" * 40)
    print(f"TOTAL TIME    : {total_duration:8.2f} s")
    print("="*40)

if __name__ == "__main__":
    main()
