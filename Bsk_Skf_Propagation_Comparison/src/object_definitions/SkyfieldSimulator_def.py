import os
import shutil # For therminal width
import logging
import numpy as np
from numpy.typing import NDArray
from typing import Optional
from datetime import datetime, timedelta, timezone

# from object_definitions.BaseSimulator_def import BaseSimulator
from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
from object_definitions.SimData_def import SimObjData, SimData
from plotting.DataLoader_def import DataLoader

from skyfield.timelib import Time
from skyfield.api import EarthSatellite, load


class SkyfieldSimulator():
    """
    =========================================================================================================
    ATTRIBUTES:
        cfg             
        startTime       
        duration       
        deltaT          
        skf_satellites
        sim_data (list[SimObjectData])
    =========================================================================================================
    """
    
    def __init__(self, cfg: Config) -> None:
        logging.debug("Setting up Skyfield simulation...")

        ###############
        # Load config #s
        ###############

        d_set = cfg        # default config
        s_set = cfg.s_set  # basilisk config


        ###################################
        # Configure simulation parameters #
        ###################################

        startTime, duration, deltaT = self.get_skf_simulation_time(cfg)


        #######################################################
        # Load TLE files to create Skyfield satellite objects #
        #######################################################

        # Create a tle file containing all the TLE information from the config file
        output_tle_path = self.create_tle_file(cfg)

        # Genera a list of Skyfield Earth satellites
        # (The notation is confusing when the other parameters unique for Skyfield don't have the prefix 'skf'.
        # This is done to distinguish from cfg.satellites in this case)
        skf_satellites = load.tle_file(output_tle_path)

        # Verify that skf_satellites has the same number of elements as 'satellites' from config
        if not len(skf_satellites) == len(cfg.satellites):
            raise ValueError(f"There is a mismatch between the number of elements in 'skf_satellites' and 'cfg.satellites'. ({len(skf_satellites)} against {len(cfg.satellites)})")


        ####################################
        # Set SkyfieldSimulator attributes #
        ####################################

        self.cfg: Config = cfg
        self.startTime: datetime = startTime
        self.duration: int = duration
        self.deltaT: int = deltaT
        self.skf_satellites: list[EarthSatellite] = skf_satellites
        self.sim_data: Optional[SimData] = None

        logging.debug("Skyfield simulation setup complete")



    def run(self) -> None:

        logging.debug("Running the Skyfield simulation...")
        
        startTime = self.startTime
        duration = self.duration
        deltaT = self.deltaT
        skf_satellites = self.skf_satellites
        satellites = self.cfg.satellites

        ########################
        # Populate time-vector #
        ########################
        times_i = []
        ts = load.timescale()
        for t in range(0, duration, deltaT):
            times_i.append(ts.utc(startTime + timedelta(seconds=t)))

        # Convert to matplotlib-compatible simulation offset
        sim_offset = self.skf_time_to_offset(times_i, startTime)
        
        
        #################################################################################
        # Run sgp4 propagation to get position and velocity at all simualtion timesteps #
        #################################################################################
        sim_data: list[SimObjData] = []

        n_sats = len(skf_satellites)
        n_t = len(times_i)
        total_steps = n_sats * n_t
        current_step = 0

        for i, skf_sat in enumerate(skf_satellites):
            # Verify that we are iterating over the satellites in the correct order
            if not isinstance(skf_sat.name, str):
                try:
                    skf_sat_name = str(skf_sat.name)
                except:
                    raise TypeError(f"skf_sat.name is of type {type(skf_sat.name)}, and couldn't be converted to str")
            else: 
                skf_sat_name = str(skf_sat.name)
            
            if not skf_sat_name == satellites[i].name:
                raise NameError(f"There is a mismatch between the skf_satellites and the cfg.satellites. Satellite nr. {i} in skf_satellites is named {skf_sat.name} while the corresponding satellkite in cfg.satellites is named {satellites[i].name}")
            
            # Pre-allocate for speed 
            positions_eci = np.empty((len(times_i), 3), dtype=float)
            velocities_eci = np.empty((len(times_i), 3), dtype=float)

            # Getting states in ECI frame
            for j, t in enumerate(times_i):
                positions_eci[j, :] = skf_sat.at(t).position.m
                velocities_eci[j, :] = skf_sat.at(t).velocity.m_per_s

                current_step += 1
                self.print_progress(current_step, total_steps)

            sim_object_data = SimObjData(
                skf_sat_name,
                sim_offset,
                positions_eci.T,
                velocities_eci.T,
            )

            sim_data.append(sim_object_data)
        
        # Set SkyfieldSimulator attribute sim_data
        self.sim_data = SimData(sim_data)

        # Write simulation data to file
        self.output_data()

        logging.debug("Skyfield simulation complete")


    def output_data(self) -> None:
        """
        Write the simulation data to a file named '<cfg.timestamp_str>_skf.h5' stored in data/sim_data/
        """
        
        # Check that simulation data has been stored
        if self.sim_data is None:
            raise ValueError("Simulation data not yet generated. Call skf.run() before skf.output_data().")
        
        # Log data to file
        self.sim_data.write_data_to_file(self.cfg.timestamp_str, "skf")


    def extract_initial_states_and_update_satellites(self, cfg: Config) -> None:
        """
        Extracts the initial states for each satellite in self.sim_data, 
        and calls a function to update their corresponding Satellite object in config.satellites
        """

        if cfg.use_old_skf_data:
            logging.debug("Skipping the Skyfield simulation. Old Skyfield simulation data will be loaded instead...")

            # Initialize data loader and processor objects
            data_loader = DataLoader()

            # Get the old skyfield datafile name 
            search_str = cfg.old_skf_data_timestamp + "_skf"
            matching_datafile_names = data_loader.get_datafiles_by_timestamp(search_str)
            if len(matching_datafile_names) > 1:
                raise ValueError(f"Found more than one match from the old skyfield data timestamp. Matches found: {matching_datafile_names}")
            elif len(matching_datafile_names) == 0:
                raise ValueError(f"Found no old sSkyfield datafile name matches for: {search_str}")
            old_skf_data_filename = matching_datafile_names[0]

            # Load old Skyfield data
            old_skf_sim_data = data_loader.load_sim_data_file(old_skf_data_filename)

            # set sim_data attribute 
            self.sim_data = old_skf_sim_data

            # Write simulation data to file
            self.output_data()


        # Verify that simulation data has been stored in self.sim_data
        if self.sim_data == None:
            raise ValueError("No simulation data has been stored in SkyfieldSimulation.sim_data")
        
        # Extract initial state for each of the satellites, and update the initial conditions for
        # their corresponding Satellite object in config.satellites
        for i, sat in enumerate(self.sim_data.sim_data):
            cfg.satellites[i].extract_initial_states_and_update(sat)





    @staticmethod
    def get_skf_simulation_time(cfg: Config) -> tuple[datetime, int, int]:
        cfg_startTime = cfg.startTime
        cfg_duration = cfg.simulationDuration
        cfg_deltaT = cfg.s_set.deltaT
        """
        Parses and converts simulation time parameters from default config into skyfield-compatible types.

        RETURNS:
            skf_startTime       (datetime) utc time
            skf_duration        (float) seconds
            skf_deltaT          (int) seconds
        """

        # Parse simulation starttime and convert into a UTC datetime object
        try:
            skf_startTime = datetime.strptime(cfg_startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
        except:
            raise ValueError("Failed to convert config parameter 'startTime' to a datetime object.")
        
        # Convert cfg_duration float(hours) -> int(seconds)
        skf_duration = int(3600 * cfg_duration) 
        if skf_duration < (3600 * cfg_duration):
            raise ValueError("Type conversion float -> int for 'skf_duration' caused a reduction in its value!")

        # deltaT
        skf_deltaT = int(cfg_deltaT)
        if skf_deltaT < cfg_deltaT:
            raise ValueError("Type conversion float -> int for 'skf_deltaT' caused a reduction in its value!")
        
        return skf_startTime, skf_duration, skf_deltaT
    
    
    @staticmethod
    def create_tle_file(cfg: Config) -> str:
        """
        Creates/overwrites a .txt file containing all satellite TLEs from the default config.
        Skyfield requries such a file for satellite object generation. 

        Returns:
            output_tle_path (str): The path to the generated tle file
        """

        logging.debug("Creating TLE .txt file from config")

        satellites = cfg.satellites
        tle_export_path = cfg.tle_export_path
        output_tle_name = "gnss_r_tle.txt"
        output_tle_path = os.path.join(tle_export_path, output_tle_name)
        
        # TODO: Check valid output_path

        try:
            # Validate or create directory specified in 'tle_export_path':
            if not os.path.isdir(tle_export_path):
                logging.info(f"Directory '{tle_export_path}' not found, attempting to create it.")
                os.makedirs(tle_export_path, exist_ok=True)

            # Write to output file:
            with open(output_tle_path, "w") as f:
                for i, sat in enumerate(satellites):
                    tle_name = sat.name
                    tle_line1 = sat.tle_line1
                    tle_line2 = sat.tle_line2

                    f.write(f"{tle_name}\n{tle_line1}\n{tle_line2}\n")

            # Validate output:
            if not os.path.isfile(output_tle_path):
                raise FileNotFoundError(f"Failed to create file at '{output_tle_path}'.")
            if os.path.getsize(output_tle_path) == 0:
                raise IOError(f"TLE file '{output_tle_path}' was created but is empty.")
            
            return output_tle_path


        except Exception as e:
            raise ValueError(f"Error while creating TLE file: {e}")


    @staticmethod
    def skf_time_to_offset(skf_time_vec: list[Time],
                           simulation_startTime: datetime
                           ) -> NDArray[np.float64]:
        """
        Converts a list of Skyfield Time objects into simulation time offsets [seconds]
        relative to the simulation start time.

        INPUTS:
            skf_time_vec (list[skyfield.timelib.Time]):
            List of Skyfield Time objects representing simulation timestamps.
            simulation_startTime (datetime):
            UTC datetime representing the start of the simulation.

        RETURNS:
            simulation_offset (np.ndarray):
            1D array of time offsets [seconds], where 0 = start time.
        """

        # Ensure start time is timezone-aware in UTC
        if simulation_startTime.tzinfo is None:
            simulation_startTime = simulation_startTime.replace(tzinfo=timezone.utc)

        # Convert Skyfield times to UTC datetimes
        utc_datetimes = [t.utc_datetime() for t in skf_time_vec]

        # Compute offset (in seconds) from simulation start
        simulation_offset = np.array([
            (dt - simulation_startTime).total_seconds() for dt in utc_datetimes
        ])

        return simulation_offset
    

    @staticmethod
    def print_progress(step: int, total: int, bar_len: int = 40) -> None:
        """
        Simple text-based progress bar.

        Parameters
        ----------
        step : int
            Current completed step (1-based).
        total : int
            Total number of steps.
        prefix : str
            Text shown before the progress bar.
        bar_len : int
            Character width of the bar.
        """
        yellow = "\033[33m"
        reset = "\033[0m"
        hide_cursor = "\033[?25l"
        show_cursor = "\033[?25h"
        prefix = "Progress: "
        cursor_hidden = False

        # Hide cursor on first call
        if not cursor_hidden:
            print(hide_cursor, end="", flush=True)
            cursor_hidden = True

        if total <= 0:
            frac = 1.0
        else:
            frac = step / total

        frac = max(0.0, min(1.0, frac))  # clamp to [0, 1]
        percent = int(frac * 100.0)

        # Terminal width
        width = shutil.get_terminal_size().columns

        # Reserve space for: prefix, ':  ', percent text '100.00%', two spaces
        reserved = len(prefix) + 2 + len("100%")  # small safety margin
        bar_len = max(10, width - reserved)  # prevent too-small bar

        # Generate bar
        filled = int(bar_len * frac)
        bar = "█" * filled

        # Color only the bar + percent in yellow
        colored_prefix = f"{yellow}{prefix}{reset}"
        colored_bar = f"{yellow}|{bar}{reset}"
        colored_percent = f"{yellow}{percent:3d}%{reset}"

        print(f"\r{colored_prefix}{colored_percent}{colored_bar} ", end="", flush=True)

        # Show cursor again at 100%
        if percent >= 100:
            print(show_cursor, end="", flush=True)
            cursor_hidden = False
            print()  # newline at end