from __init__ import initialize
from constants import MC_CONFIG_PATH, BASE_CONFIG_PATH
from object_definitions.Config_def import Config
from object_definitions.MonteCarloConfig_def import MonteCarloConfig
from object_definitions.BasiliskSimulator_def import BasiliskSimulator


def simulate_single_gnssr_mission(mc_cfg: MonteCarloConfig, run_idx: int = 0):
    
    # Load config and resolve overrides (if necessary)
    cfg = Config(BASE_CONFIG_PATH, mc_cfg, run_idx)

    # Initialize Basilisk simulator 
    bsk = BasiliskSimulator(cfg)

    # Run Basilisk simulator for the configured scenario
    bsk.run()

    # Output data to file
    bsk.output_data()



def monte_carlo_gnssr_mission(mc_cfg: MonteCarloConfig): 
    # run 'n' bsk simulations
    for i in range(mc_cfg.num_bsk_sims):
        simulate_single_gnssr_mission(mc_cfg, i)



if __name__ == "__main__":
    
    # Initialize monte carlo config instance and set up logging
    mc_cfg = initialize(MC_CONFIG_PATH)
    
    # If Monte Carlo is enabled, generate 'n-1' override files and run 'n' bsk simulations
    if mc_cfg.mc_enabled:
        mc_cfg.generate_config_overrides()
        monte_carlo_gnssr_mission(mc_cfg)
    
    # Else run a single bsk simulation
    else:
        simulate_single_gnssr_mission(mc_cfg)


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