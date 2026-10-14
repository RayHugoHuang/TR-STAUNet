import os, sys, subprocess

def install_torch():
    """
    Manually install PyTorch with specific CUDA version.
    Command provided by user for env: CUDA 12.9, Python 3.9
    """
    print("\n--- Installing PyTorch (CUDA 12.9) ---")
    torch_cmd = [
        sys.executable, "-m", "pip", "install",
        "torch==2.8.0", "torchvision==0.23.0", "torchaudio==2.8.0", 
        "--index-url", "https://download.pytorch.org/whl/cu129"
    ]
    
    try: # The executed instructions are displayed to the user.
        print(f"Command: {' '.join(torch_cmd)}")
        subprocess.check_call(torch_cmd)
        print("✔ PyTorch installed successfully.")
    except subprocess.CalledProcessError:
        print("❌ Failed to install PyTorch.")
        print("Please try running the command manually.")
        sys.exit(1)

def init_project():
    # 1. Create Directories
    folders = [
        "results",
        "weights",
        os.path.join("dataset_storage", "raw"),
        os.path.join("dataset_storage", "processed"),
        os.path.join("dataset_storage", "raw", "gt_data")
    ]
    
    print("--- Creating Directories ---")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"Checked: {folder}")

    # 2. Install Requirements
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print("\n--- Installing Dependencies ---")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("Dependencies installed.")
        except subprocess.CalledProcessError:
            print("Installation failed.")
    else:
        print(f"\nWarning: {req_file} not found.")
    
    # 3. Install torch
    install_torch()

if __name__ == "__main__":
    init_project()