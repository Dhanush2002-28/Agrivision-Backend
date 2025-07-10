#!/usr/bin/env python3
"""
Quick setup script to install required packages for model comparison
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    """Install all required packages"""
    packages = [
        "matplotlib",
        "seaborn", 
        "xgboost",
        "scikit-learn>=1.0.0"
    ]
    
    print("🔧 Installing required packages for model comparison...")
    print("=" * 50)
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("=" * 50)
    print(f"✅ Installation complete: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("🚀 Ready to run model comparison!")
        print("Run: python compare_models.py")
    else:
        print("⚠️ Some packages failed to install. Please install manually.")

if __name__ == "__main__":
    main()
