import os
import sys
import importlib

# Fix Windows console unicode issues
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # In case stdout doesn't support reconfiguration in some environments

def check_python_version():
    """Checks if the Python version is 3.8 or higher."""
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major == 3 and minor >= 8:
        print(f"✅ Python {version_str}")
        return True
    else:
        print(f"❌ Python {version_str} (Requires 3.8+)")
        return False

def check_pytorch():
    """Checks if PyTorch is installed and checks for CUDA/GPU support."""
    try:
        import torch
        pytorch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✅ PyTorch {pytorch_version} with CUDA")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU: {gpu_name}")
        else:
            print(f"✅ PyTorch {pytorch_version} (CPU only - Training will be slow)")
            print("⚠️  No GPU found. If you have an NVIDIA GPU, make sure CUDA Toolkit and appropriate PyTorch are installed.")
        return True
    except ImportError:
        print("❌ PyTorch is NOT installed!")
        return False

def check_packages():
    """Verifies that all required third-party packages can be imported successfully."""
    packages = {
        'ultralytics': 'Ultralytics (YOLOv8)',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'yaml': 'PyYAML'
    }
    
    all_success = True
    for pkg_name, display_name in packages.items():
        try:
            pkg = importlib.import_module(pkg_name)
            # Retrieve version if available
            version = getattr(pkg, '__version__', 'unknown')
            print(f"✅ {display_name} {version}")
        except ImportError:
            print(f"❌ {display_name} is NOT installed!")
            all_success = False
            
    if all_success:
        print("✅ All packages installed successfully!")
    return all_success

def create_folders():
    """Creates the required project directories automatically."""
    folders = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "models/yolov8/weights",
        "models/yolov8/results",
        "results/detections",
        "src"
    ]
    
    # We want to create directories relative to the project root (where this setup script is run from)
    created_count = 0
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            created_count += 1
            
    print(f"📁 Project folders verified/created ({created_count} new folders created).")

def main():
    print("🔍 Checking Environment Setup...")
    print("-" * 40)
    
    python_ok = check_python_version()
    pytorch_ok = check_pytorch()
    packages_ok = check_packages()
    
    print("-" * 40)
    create_folders()
    
    if python_ok and pytorch_ok and packages_ok:
        print("\n🎉 Environment is fully configured and ready for pothole detection training!")
        sys.exit(0)
    else:
        print("\n❌ Environment check failed. Please resolve the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
