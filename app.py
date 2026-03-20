"""
Hugging Face Spaces entry point.
This file is required by HF Spaces (Streamlit SDK) — it simply runs the dashboard.
"""
import sys
import os

# Ensure evalflow is importable when running from HF Spaces
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the dashboard
from dashboard_app import main

if __name__ == "__main__":
    main()
