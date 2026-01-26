import logging

from __init__ import initialize
from plotting.plot import plot
from object_definitions.BasiliskSimulator_def import BasiliskSimulator
from object_definitions.SkyfieldSimulator_def import SkyfieldSimulator

def simualte_satellite_orbits():

    # Load config and define all neccessary objects
    cfg = initialize('Bsk_Skf_Propagation_Comparison/configs/default.yaml')

    # Bypass the simulation if we instead want to plot old datafiles
    if not cfg.bypass_sim_to_plot:
    
        # Initialize Skyfield SGP4 Propagator
        skf = SkyfieldSimulator(cfg)

        # Bypass Skyfield simulation if old data should be used instead
        if not cfg.use_old_skf_data:

            # Run Skyfield SGP4 propagator
            skf.run()

        # Extract initial states @ simulation startTime and update cfg.satellites
        skf.extract_initial_states_and_update_satellites(cfg)

        # Initialize Basilisk Dynamic Model Propagator
        bsk = BasiliskSimulator(cfg)

        # Run Basilisk Dynamic Model Propagator
        bsk.run()

    # Plot results
    plot(cfg)


if __name__ == "__main__":
    simualte_satellite_orbits()


# TODO
"""
* Merge config files into one
* Fix the creation of common Satellite objects that are compatible with both the new config structure and TLE-handeling


Simulator Misk:
* To Master: Enable the option to queue multiple simulation runs with different configurations
"""