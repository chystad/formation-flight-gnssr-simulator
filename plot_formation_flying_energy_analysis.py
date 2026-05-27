from pathlib import Path
import sys
import runpy

# Add Formation_Flying_Energy_Analysis/src to Python's import search path
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "Formation_Flying_Energy_Analysis" / "src"
sys.path.insert(0, str(SRC_DIR))

runpy.run_path(str(SRC_DIR / "plotting" / "optimized_plotting_from_file.py"))