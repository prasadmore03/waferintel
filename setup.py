#!/usr/bin/env python3
"""
WaferIntel Setup Script
Quick installation and setup for WaferIntel project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_rag.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_files():
    """Check if all required files are present"""
    required_files = [
        "run_rag_app.py",
        "main_app_rag.py", 
        "rag_wafer_assistant.py",
        "requirements_rag.txt"
    ]
    
    required_models = [
        "models/wafer_model.h5",
        "models/external_wafer_model.h5",
        "models/wafer_class_names.npy",
        "models/external_wafer_classes.npy"
    ]
    
    print("🔍 Checking required files...")
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    for model in required_models:
        if not os.path.exists(model):
            missing_files.append(model)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files present!")
        return True

def main():
    """Main setup function"""
    print("🚀 WaferIntel Setup")
    print("=" * 50)
    
    # Check files
    if not check_files():
        print("\n❌ Setup failed: Missing required files")
        return
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed: Could not install requirements")
        return
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n🎯 To run WaferIntel:")
    print("   python run_rag_app.py")
    print("\n🌐 Then open: http://localhost:8502")
    print("=" * 50)

if __name__ == "__main__":
    main()
