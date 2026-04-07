#!/usr/bin/env python3
"""
Launcher for AI-Powered Wafer Defect Detection + RAG Assistant
Fixes PATH issues and runs the RAG Streamlit app
"""

import sys
import os
import subprocess

def main():
    """Main launcher function"""
    print("🧠 AI-Powered Wafer Defect Detection + RAG Assistant")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("main_app_rag.py"):
        print("❌ Error: main_app_rag.py not found!")
        print("Please run this script from the project directory.")
        return
    
    # Check models directory
    if not os.path.exists("models"):
        print("⚠️ Warning: models/ directory not found")
        print("Please copy your trained models to the models/ directory")
    
    # Check RAG assistant file
    if not os.path.exists("rag_wafer_assistant.py"):
        print("❌ Error: rag_wafer_assistant.py not found!")
        return
    
    print("📋 Starting RAG-powered Streamlit app...")
    print("🌐 App will open in your browser at: http://localhost:8503")
    print("⏹️ Press Ctrl+C to stop the app")
    print("🤖 RAG Assistant will provide detailed defect explanations")
    print("=" * 60)
    
    try:
        # Try running with python -m streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "main_app_rag.py", 
            "--server.port", "8503",
            "--server.headless", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 RAG app stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("💡 Try running manually: python -m streamlit run main_app_rag.py")
        print("🔧 Make sure all dependencies are installed: pip install -r requirements_rag.txt")

if __name__ == "__main__":
    main()
