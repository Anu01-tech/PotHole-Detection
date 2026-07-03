"""
setup_environment.py

This script automates the environment setup for the Pothole Detection project.
It performs the following steps:
1. Creates the complete project folder structure.
2. Checks for python version compatibility.
3. Automatically installs the required packages from requirements.txt.
4. Checks for PyTorch installation and NVIDIA CUDA (GPU) availability.
5. Verifies that all key packages (Ultralytics, OpenCV, etc.) can be successfully imported.

Usage:
    python src/setup_environment.py
"""

import os
import sys
import subprocess
import platform

def print_separator():
    print("=" * 60)

def create_directory_structure():
    """Creates the standard directory tree required for this project."""
    print("\n[1/5] Creating Project Directory Structure...")
    
    # Define directories to create
    directories = [
        # Dataset directories (YOLO format)
        os.path.join("dataset", "images", "train"),
        os.path.join("dataset", "images", "val"),
        os.path.join("dataset", "images", "test"),
        os.path.join("dataset", "labels", "train"),
        os.path.join("dataset", "labels", "val"),
        os.path.join("dataset", "labels", "test"),
        
        # Model storage and results
        os.path.join("models", "yolov11", "weights"),
        os.path.join("models", "yolov11", "results"),
        os.path.join("models", "yolov8", "weights"),
        os.path.join("models", "yolov8", "results"),
        
        # Source code & App directories
        "src",
        "app",
        
        # Results visualizations
        os.path.join("results", "comparison_charts"),
        os.path.join("results", "detection_examples"),
        
        # Documentation
        "docs"
    ]
    
    # Create each directory
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"  + Created directory: {directory}")
        else:
            print(f"  o Directory already exists: {directory}")
            
    print("Directory structure setup completed successfully!")

def get_system_info():
    """Gathers and displays information about the operating system and python environment."""
    print_separator()
    print("SYSTEM INFORMATION:")
    print(f"  OS: {platform.system()} {platform.release()} (Version: {platform.version()})")
    print(f"  Architecture: {platform.machine()}")
    print(f"  Python Version: {sys.version}")
    print_separator()

def install_requirements():
    """Installs dependencies listed in requirements.txt."""
    print("\n[2/5] Installing Dependencies from requirements.txt...")
    req_file = "requirements.txt"
    
    if not os.path.exists(req_file):
        print(f"  [ERROR] {req_file} not found in current directory: {os.getcwd()}")
        print("  Please make sure requirements.txt is in the project root directory.")
        sys.exit(1)
        
    try:
        # We run the pip install command using subprocess
        print(f"  Running: pip install -r {req_file} ...")
        # Using sys.executable ensures we use the pip of the current python environment
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("  Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Installation failed with exit code {e.returncode}.")
        print("  Please try installing packages manually using: pip install -r requirements.txt")
        sys.exit(1)

def check_pytorch_and_cuda():
    """
    Checks if PyTorch is installed and whether it can access the NVIDIA CUDA GPU.
    CUDA is NVIDIA's parallel computing platform that allows PyTorch to run calculations 
    on the GPU, making training up to 10-50x faster than on a CPU.
    """
    print_separator()
    print("[3/5] Checking PyTorch and CUDA (GPU) Status...")
    
    try:
        import torch
        print(f"  PyTorch Version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"  CUDA GPU Available: {cuda_available}")
        
        if cuda_available:
            print(f"  CUDA Device Count: {torch.cuda.device_count()}")
            print(f"  CUDA Device Name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print("\n  [SUCCESS] GPU training is fully supported and enabled on your machine!")
        else:
            print("\n  [WARNING] CUDA is not available or not detected by PyTorch.")
            print("  Training will run on the CPU, which is significantly slower.")
            print("  If you have an NVIDIA GPU, follow these steps to enable CUDA support:")
            print("  1. Check if NVIDIA drivers are up to date.")
            print("  2. Uninstall torch/torchvision: pip uninstall torch torchvision -y")
            print("  3. Install PyTorch with CUDA using the official command from pytorch.org:")
            print("     e.g., pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            
    except ImportError:
        print("  [ERROR] PyTorch (torch) is not installed. Please run installation step first.")

def verify_installations():
    """Imports each required package to verify they are correctly installed and ready for use."""
    print_separator()
    print("[4/5] Verifying Package Imports...")
    
    packages = {
        'ultralytics': 'Ultralytics (YOLOv8 & YOLOv11)',
        'cv2': 'OpenCV (Computer Vision)',
        'numpy': 'NumPy (Numerical computations)',
        'pandas': 'Pandas (Data analysis)',
        'matplotlib': 'Matplotlib (Charts & Plotting)',
        'seaborn': 'Seaborn (Stylized visualizations)',
        'streamlit': 'Streamlit (Web App framework)'
    }
    
    success = True
    for package, name in packages.items():
        try:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [PASS] {name:<30} | Version: {version}")
        except ImportError as e:
            print(f"  [FAIL] {name:<30} | Error: {e}")
            success = False
            
    if success:
        print("\n  [SUCCESS] All package verifications passed!")
    else:
        print("\n  [ERROR] Some package verifications failed. Check errors above.")
        
    return success

def main():
    print_separator()
    print("          POTHOLE DETECTION PROJECT: SETUP ENVIRONMENT")
    print_separator()
    
    # 1. System Info
    get_system_info()
    
    # 2. Directory structure
    create_directory_structure()
    
    # 3. Install packages
    install_requirements()
    
    # 4. Check CUDA GPU
    check_pytorch_and_cuda()
    
    # 5. Verify Package Imports
    success = verify_installations()
    
    print_separator()
    if success:
        print("[5/5] ENVIRONMENT SETUP COMPLETE!")
        print("You are ready to proceed to Phase 2 (Dataset Preparation).")
    else:
        print("[5/5] SETUP FINISHED WITH ERRORS.")
        print("Please resolve the package import failures before proceeding.")
    print_separator()

if __name__ == "__main__":
    main()
