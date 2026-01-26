from pathlib import Path
import sys

# Add Propagation_Comparison/src to Python's import search path
THIS_DIR = Path(__file__).resolve().parent
PC_SRC = THIS_DIR / "Formation_Flying_Energy_Analysis" / "src"
sys.path.insert(0, str(PC_SRC))