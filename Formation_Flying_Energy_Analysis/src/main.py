import logging

from __init__ import initialize
from object_definitions.BasiliskSimulator_def import BasiliskSimulator

def simulate_gnssr_mission():

    # Load config and define all neccessary objects
    cfg = initialize('Formation_Flying_Energy_Analysis/configs/base.yaml')

    # Initialize Basilisk Dynamic Model Propagator
    bsk = BasiliskSimulator(cfg)

    # Run Basilisk Dynamic Model Propagator
    bsk.run()


if __name__ == "__main__":
    simulate_gnssr_mission()


# TODO
"""
* [Optional] Merge config files into one 
* [Optional] Optimize the Skyfield simulation by editing SkyfieldSimulator.run() to call sat.at(times_segment) in a batch 
* Make Basilisk model parameters part of the Basilisk config (was talking about exp atm scale height, for example)
* [IMPORTANT] Get MSIS model parameters from data, and update during runtime

Simulator Misk:
* To Master: Enable the option to queue multiple simulation runs with different configurations
"""