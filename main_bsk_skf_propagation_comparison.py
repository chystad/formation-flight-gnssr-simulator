from pathlib import Path
import sys

# Add Propagation_Comparison/src to Python's import search path
THIS_DIR = Path(__file__).resolve().parent
PC_SRC = THIS_DIR / "Bsk_Skf_Propagation_Comparison" / "src"
sys.path.insert(0, str(PC_SRC))

from Bsk_Skf_Propagation_Comparison.src.main import simualte_satellite_orbits
simualte_satellite_orbits()