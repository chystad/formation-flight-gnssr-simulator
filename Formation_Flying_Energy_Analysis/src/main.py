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
* [IMPORTANT] Don't use cannonball SRP and Drag effector. This makes them attitude independent!
* Enable the option to queue multiple simulation runs with different configurations
* Solar panel feature:
    - From config 'SP_PARAMETERS', calculate which face has the largest solar panel area, and then set r_PB_B based on this.
    - Make the guidance system use r_PB_B dynamically (right now it is just hard-coded that the face with largest panel area
    is the Z+ face)
* Remove option to select custom initial state vector (sat_init_source and init_state_vec from config)
* Make FSW subscribe to battery state and use it in state machine
* Instead of storing spacecraft specific data as lists stored as attributes, store them as a per-satellite bundle (dataclass)
* [IMPORTANT] Make a robust and easilly expandable system for writing simulation results to file
"""