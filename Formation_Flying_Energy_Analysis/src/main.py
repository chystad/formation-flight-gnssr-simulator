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
* Move GNC module from 'BasiliskSimulator_def' into its own object def script and schedule it as a task.
* Enable the option to queue multiple simulation runs with different configurations
"""