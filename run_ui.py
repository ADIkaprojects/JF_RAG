import subprocess
import os
import sys
from pathlib import Path

def main():
    """Start the Streamlit UI"""
    project_root = Path(__file__).parent.absolute()
    venv_python = project_root / 'venv' / 'Scripts' / 'python.exe'
    if not venv_python.exists():
        print(" Virtual environment not found!")
        print(f"Please create it with: python -m venv venv")
        sys.exit(1)
    
    print("=" * 60)
    print("MultiModal RAG Chatbot - Quick Start")
    print("=" * 60)
    print()
    print("Make sure these are running:")
    print("  1. Ollama: ollama serve")
    print("  2. API: python -m uvicorn api.main:app --port 8000")
    print()
    print("Starting Streamlit UI on http://localhost:8501...")
    print()
    
    streamlit_path = venv_python.parent / 'streamlit.exe'
    
    subprocess.run([
        str(streamlit_path),
        'run',
        str(project_root / 'ui' / 'app.py'),
        '--logger.level=info'
    ])

if __name__ == '__main__':
    main()
