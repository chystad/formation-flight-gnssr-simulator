import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import PngImagePlugin
from pathlib import Path
import cartopy.crs as ccrs
import pymap3d as pm
from datetime import datetime, timedelta, timezone

from object_definitions.Config_def import Config
from object_definitions.SimData_def import SimData, RelObjData
from plotting.DataLoader_def import DataLoader
from plotting.DataProcessor_def import DataProcessor


RTN_DOWNSAMP_FAC: int = 5
PLT_SAVE_FOLDER_PATH = Path('output_data/sim_plt')
PLT_HEIGHT = 6.0
PLT_WIDTH = 16.0


def plot(cfg: Config) -> None:
    """
    Description coming here...
    
    Args:
        cfg (Config): Simulation config
    """
    # plot() does nothing if show_plots == false and save_plots == false.
    # Exit if that's the case
    if (not cfg.show_plots) and (not cfg.save_plots):
        logging.debug("Config settings specify: show_plots == false and save_plots == false, which makes the plotting function obsolete. -> Exiting plot()...")
        return

    # Stop plots to clutter the terminal with debug info
    quiet_plots()
    
    ##################################
    # Fetch and Load simulation data #
    ##################################

    # Initialize data loader and processor objects
    data_loader = DataLoader()

    if not cfg.bypass_sim_to_plot:
        # plot results from this simulation 
        data_timestamp = cfg.timestamp_str  
    else:
        # plot results from a previous simulation
        data_timestamp = cfg.data_timestamp_to_plot

    # Get all datafiles with the corresponding timestamp
    datafiles_to_plot = data_loader.get_datafiles_by_timestamp(data_timestamp)
    
    # Load simulation data
    skf_sim_data, bsk_sim_data = data_loader.load_and_separate_data(datafiles_to_plot)
    
    ############
    # Plotting #
    ############

    # plot_groundtrack_comparison_start_stop(cfg, skf_sim_data, bsk_sim_data, 160, 168)

    # plot_groundtrack_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_pos_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_simulator_state_diff(cfg, skf_sim_data, bsk_sim_data)
    
    # plot_rel_pos_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_simulator_rel_state_diff(cfg, skf_sim_data, bsk_sim_data)

    # plot_altitude_comparison(cfg, skf_sim_data, bsk_sim_data)

    # plot_simulator_state_abs_diff(cfg, skf_sim_data, bsk_sim_data)


    

    #####################
    # Experiment 1 data #
    #####################
    # bsk_data_timestamps = [("20251216_084208", "Bsk: RKF78, All")]
    #                        #("20251216_092602", "Bsk: RK4, All")]
    # base_label = "Skf: SGP4 (base)"
    # skf_base_data_timestamp = "20251216_073738"


    #####################
    # Experiment 2 data #
    #####################
    # # For spherical harmonics only against Skyfield
    # bsk_data_timestamps = [("20251216_110041", "Bsk: RKF78, All"),
    #                        ("20251216_110616", "Bsk: RKF78, SH2"),
    #                        ("20251216_110859", "Bsk: RKF78, SH3"),
    #                        ("20251216_111031", "Bsk: RKF78, SH4"),]
    #                     #    ("20251216_111211", "Bsk: RKF78, SH4 + Drag"),
    #                     #    ("20251216_111309", "Bsk: RKF78, SH4 + SRP"),
    #                     #    ("20251216_111726", "Bsk: RKF78, SH4 + Sun"),
    #                     #    ("20251216_111818", "Bsk: RKF78, SH4 + Moon")]
    # base_label = "Skf: SGP4 (base)"
    # skf_base_data_timestamp = cfg.old_skf_data_timestamp

    # # For different combinations of 2nd order spherical harmonics and other perturbations
    # bsk_data_timestamps = [
    #                     #    ("20251216_110616", "Bsk: RKF78, SH2"),
    #                     #    ("20251216_113404", "Bsk: RKF78, SH2 + Drag"),
    #                     #    ("20251216_113459", "Bsk: RKF78, SH2 + SRP"),
    #                     #    ("20251216_113555", "Bsk: RKF78, SH2 + Sun"),
    #                     #    ("20251216_113645", "Bsk: RKF78, SH2 + Moon"),
    #                     #    ("20251216_114138", "Bsk: RKF78, SH2 + Rest"),
    #                        ("20251216_111211", "Bsk: RKF78, SH4 + Drag"),
    #                        ("20251216_111309", "Bsk: RKF78, SH4 + SRP"),
    #                        ("20251216_111726", "Bsk: RKF78, SH4 + Sun"),
    #                        ("20251216_111818", "Bsk: RKF78, SH4 + Moon"),
    #                        ("20251216_110041", "Bsk: RKF78, All")]
    # base_label = "Bsk: RKF78, SH4 (base)"
    # skf_base_data_timestamp = "20251216_111031" # REMEMBER TO SELECT THE CORRECT STRING-ENDING IN PLOTTING FUNCTIONS!!!


    # # Total combination of all perturbations:
    # bsk_data_timestamps = [
    #                        ("20251216_110616", "Bsk: RKF78, SH2"),
    #                        ("20251216_110859", "Bsk: RKF78, SH3"),
    #                        ("20251216_111031", "Bsk: RKF78, SH4"),
    #                     #    ("20251216_111211", "Bsk: RKF78, SH4 + Drag"),
    #                     #    ("20251216_111309", "Bsk: RKF78, SH4 + SRP"),
    #                     #    ("20251216_111726", "Bsk: RKF78, SH4 + Sun"),
    #                     #    ("20251216_111818", "Bsk: RKF78, SH4 + Moon"),
    #                        ("20251216_110041", "Bsk: RKF78, All")]
    # base_label = "Skf: SGP4 (base)"
    # skf_base_data_timestamp = cfg.old_skf_data_timestamp # REMEMBER TO SELECT THE CORRECT STRING-ENDING IN PLOTTING FUNCTIONS!!!
    # base_label = "Bsk: RKF78, SH4 (base)"
    # skf_base_data_timestamp = "20251216_111031" 
    # base_label = "Bsk: RKF78, SH2 (base)"
    # skf_base_data_timestamp = "20251216_110616"


    #####################
    # Experiment 3 data #
    #####################
    # # For different integrator configurations
    # bsk_data_timestamps = [("20251216_150753", "Bsk: RK4, dt=50, All"),
    #                        ("20251216_150708", "Bsk: RK4, dt=20, All"),
    #                        ("20251216_150600", "Bsk: RK4, dt=5, All"),
    #                        ("20251216_150328", "Bsk: RK4, dt=1, All"),
    #                        ("20251216_150937", "Bsk: RKF45, All"),
    #                        ("20251216_110041", "Bsk: RKF78, All"),]
    # base_label = "Skf: SGP4 (base)"
    # skf_base_data_timestamp = cfg.old_skf_data_timestamp
   

    ###############################
    # Experiment combination data #
    ###############################
    # # For spherical harmonics only against Skyfield
    # bsk_data_timestamps = [("20251216_110616", "Bsk: RKF78, SH2"),
    #                        ("20251216_110859", "Bsk: RKF78, SH3"),
    #                        ("20251216_111031", "Bsk: RKF78, SH4"),
    #                        ("20251216_111211", "Bsk: RKF78, SH4 + Drag"),
    #                        ("20251216_111309", "Bsk: RKF78, SH4 + SRP"),
    #                        ("20251216_111726", "Bsk: RKF78, SH4 + Sun"),
    #                        ("20251216_111818", "Bsk: RKF78, SH4 + Moon"),
    #                        ("20251216_110041", "Bsk: RKF78, All")]
    # base_label = "Skf: SGP4 (base)"
    # skf_base_data_timestamp = cfg.old_skf_data_timestamp

    
    # plot_rel_pos_multi_sim_diff(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)
    # plot_rel_pos_multi_sim_diff_no_radial(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)
    # plot_rel_vel_multi_sim_diff(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)
    # plot_multi_sim_pos_vel_sim_diff_mag(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)
    # plot_alt_multi_sim_diff(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)
    # plot_simulator_state_mag_multi_sim_diff(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)
    # plot_groundtrack_multi_sim_comparison_start_stop(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)
    # plot_multi_sim_pos_sim_diff_mag(cfg, skf_base_data_timestamp, bsk_data_timestamps, base_label)

#################################
# Plotting function definitions #
#################################

def plot_groundtrack_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot the ground-track (projected trajectory on Earth) for each satellite,
    comparing Skyfield (red) and Basilisk (green).

    Positions are given in ECI and converted to geodetic (lat, lon) using pymap3d.eci2geodetic.
    One map figure is generated per satellite.
    """
    main_plt_identifier = "GroundTrack"  # Used in saved plot name

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Epoch for t = 0 (assumed to live in cfg; adjust if you store it elsewhere)
    epoch, duration, deltaT = get_simulation_time(cfg) 

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Shape sanity checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D
        t_skf_sec = np.ravel(skf.time)
        t_bsk_sec = np.ravel(bsk.time)

        # Convert to datetime arrays for pymap3d
        t_skf_dt = [epoch + timedelta(seconds=float(t)) for t in t_skf_sec]
        t_bsk_dt = [epoch + timedelta(seconds=float(t)) for t in t_bsk_sec]

        # Unpack ECI position components
        x_skf, y_skf, z_skf = skf.pos[0, :], skf.pos[1, :], skf.pos[2, :]
        x_bsk, y_bsk, z_bsk = bsk.pos[0, :], bsk.pos[1, :], bsk.pos[2, :]

        # ECI -> geodetic (lat [deg], lon [deg], alt [m])
        # pymap3d will broadcast over arrays
        lat_skf, lon_skf, _ = pm.eci2geodetic(x_skf, y_skf, z_skf, t_skf_dt)
        lat_bsk, lon_bsk, _ = pm.eci2geodetic(x_bsk, y_bsk, z_bsk, t_bsk_dt)

        # Create map figure for this satellite
        fig = plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Background: simple stock image + coastlines
        ax.stock_img()
        ax.coastlines()
        ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

        # Plot ground tracks
        sat_name = (
            skf.satellite_name
            if getattr(skf, "satellite_name", None)
            else f"Satellite {i+1}"
        )

        # Skyfield: red
        ax.plot(
            lon_skf,
            lat_skf,
            color="red",
            linewidth=1.5,
            linestyle="-",
            label=f"SKF {sat_name}",
            transform=ccrs.Geodetic(),
        )

        # Basilisk: green
        ax.plot(
            lon_bsk,
            lat_bsk,
            color="green",
            linewidth=1.5,
            linestyle=":",
            label=f"BSK {sat_name}",
            transform=ccrs.Geodetic(),
        )

        ax.set_title(f"Ground track comparison — {sat_name}")
        ax.legend(loc="lower left")

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_groundtrack_comparison_start_stop(
    cfg: Config,
    skf_sim_data: SimData,
    bsk_sim_data: SimData,
    start_plot_time_hours: float = 0.0,
    end_plot_time_hours=None,
    view_lon_min: float = 3.0,
    view_lon_max: float = 33.0,
    view_lat_min: float = 56.0,
    view_lat_max: float = 73.0,
) -> None:
    """
    Plot the ground-track (projected trajectory on Earth) for each satellite,
    comparing Skyfield and Basilisk.

    Adds:
      - start_plot_time_hours / end_plot_time_hours: limits data plotted in time
      - view_lon_min/max and view_lat_min/max: limits the visible map window

    Defaults for view window are chosen to focus on Norway + nearby region.
    """
    main_plt_identifier = "GroundTrack"  # Used in saved plot name

    if end_plot_time_hours is None:
        end_plot_time_hours = cfg.simulationDuration

    if end_plot_time_hours < start_plot_time_hours:
        raise ValueError(
            f"end_plot_time_hours ({end_plot_time_hours}) "
            f"must be >= start_plot_time_hours ({start_plot_time_hours})."
        )

    if view_lon_max <= view_lon_min or view_lat_max <= view_lat_min:
        raise ValueError(
            "Invalid map view window: require view_lon_max > view_lon_min and view_lat_max > view_lat_min."
        )

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Epoch for t = 0
    epoch, duration, deltaT = get_simulation_time(cfg)

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Shape sanity checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D [s]
        t_skf_sec = np.ravel(skf.time)
        t_bsk_sec = np.ravel(bsk.time)

        # Convert to hours for masking
        t_skf_hours = t_skf_sec / 3600.0
        t_bsk_hours = t_bsk_sec / 3600.0

        # Build masks for the requested time window
        skf_mask = (
            (t_skf_hours >= start_plot_time_hours)
            & (t_skf_hours <= end_plot_time_hours)
        )
        bsk_mask = (
            (t_bsk_hours >= start_plot_time_hours)
            & (t_bsk_hours <= end_plot_time_hours)
        )

        # If no points in range, skip this satellite
        if not np.any(skf_mask) and not np.any(bsk_mask):
            continue

        # Apply masks
        t_skf_sec_sel = t_skf_sec[skf_mask]
        t_bsk_sec_sel = t_bsk_sec[bsk_mask]

        x_skf = skf.pos[0, skf_mask]
        y_skf = skf.pos[1, skf_mask]
        z_skf = skf.pos[2, skf_mask]

        x_bsk = bsk.pos[0, bsk_mask]
        y_bsk = bsk.pos[1, bsk_mask]
        z_bsk = bsk.pos[2, bsk_mask]

        # Convert to datetime arrays for pymap3d
        t_skf_dt = [epoch + timedelta(seconds=float(t)) for t in t_skf_sec_sel]
        t_bsk_dt = [epoch + timedelta(seconds=float(t)) for t in t_bsk_sec_sel]

        # ECI -> geodetic (lat [deg], lon [deg], alt [m])
        lat_skf, lon_skf, _ = pm.eci2geodetic(x_skf, y_skf, z_skf, t_skf_dt)
        lat_bsk, lon_bsk, _ = pm.eci2geodetic(x_bsk, y_bsk, z_bsk, t_bsk_dt)

        # Create map figure for this satellite
        fig = plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # Limit the visible map window (PlateCarree expects lon/lat in degrees)
        ax.set_extent(
            [view_lon_min, view_lon_max, view_lat_min, view_lat_max],
            crs=ccrs.PlateCarree(),
        )

        # Background + details
        ax.stock_img()
        ax.coastlines()
        ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

        sat_name = (
            skf.satellite_name
            if getattr(skf, "satellite_name", None)
            else f"Satellite {i+1}"
        )

        # Skyfield
        if len(lon_skf) > 0:
            ax.plot(
                lon_skf,
                lat_skf,
                linewidth=1.5,
                linestyle="-",
                label=f"SKF {sat_name}",
                transform=ccrs.Geodetic(),
            )

        # Basilisk (your current code uses solid; change to ":" if you want dotted)
        if len(lon_bsk) > 0:
            ax.plot(
                lon_bsk,
                lat_bsk,
                linewidth=1.5,
                linestyle="-",
                label=f"BSK {sat_name}, RKF78, All",
                transform=ccrs.Geodetic(),
            )

        ax.set_title(
            f"Ground track comparison — {sat_name}\n"
            f"t ∈ [{start_plot_time_hours:.2f}, {end_plot_time_hours:.2f}] h"
        )
        ax.legend(loc="lower left")

        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_pos_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot position vectors for each satellite from two SimData datasets.

    For satellite i (0..n-1), creates a new figure and plots:
      - SKF position components (x,y,z) as different blue shades
      - BSK position components (x,y,z) as different green shades
    The time axis comes from each dataset's own `time` array.

    Args:
        skf_sim_data (SimData):
        bsk_sim_data (SimData):
    Returns:
        None
    """
    main_plt_identifier = "PosComp" # Used in the saved plot name
    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Color palettes (light → dark)
    skf_colors = ["#9ecae1", "#3182bd", "#08519c"]  # blues
    bsk_colors = ["#a1d99b", "#31a354", "#006d2c"]  # greens
    comp_labels = ["x", "y", "z"]

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape sanity checks (won't fail if shapes are as specified)
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D
        t_skf = np.ravel(skf.time) / (60*60) # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60*60) # Time [hours]

        # Create a new figure for this satellite; no explicit numbering to avoid conflicts
        plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax = plt.gca()

        # Plot SKF (blue shades)
        for comp in range(3):
            ax.plot(
                t_skf,
                skf.pos[comp, :],
                label=f"SKF {comp_labels[comp]}",
                linewidth=1.8,
                color=skf_colors[comp],
            )

        # Plot BSK (green shades)
        for comp in range(3):
            ax.plot(
                t_bsk,
                bsk.pos[comp, :],
                label=f"BSK {comp_labels[comp]}",
                linewidth=1.8,
                linestyle="--",
                color=bsk_colors[comp],
            )

        sat_name = skf.satellite_name if getattr(skf, "satellite_name", None) else f"Satellite {i+1}"
        ax.set_title(f"Position comparison — {sat_name}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("ECI Position (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_rel_pos_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot the position vectors relative to the formation chief satellite expressed in the RTN frame.

    For satellite i (0..n-1), creates a new figure and plots:
      - SKF position components (x,y,z) as different blue shades
      - BSK position components (x,y,z) as different green shades
    The time axis comes from each dataset's own `time` array.

    Args:
        skf_sim_data (SimData):
        bsk_sim_data (SimData):
    Returns:
        None
    """
    main_plt_identifier = "RelPosComp"

    # Initialize data processor
    data_processor = DataProcessor()

    # Calculate the relative position vectors from every follower satellites 
    # to the chief satellite expressed in RTN frame, and set results in rel_data attribute
    data_processor.calculate_relative_formation_movement_rtc(skf_sim_data, RTN_DOWNSAMP_FAC)
    data_processor.calculate_relative_formation_movement_rtc(bsk_sim_data, RTN_DOWNSAMP_FAC)

    skf_list = skf_sim_data.rel_data
    bsk_list = bsk_sim_data.rel_data

    assert skf_list is not None and bsk_list is not None

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Color palettes (light → dark)
    skf_colors = ["#9ecae1", "#3182bd", "#08519c"]  # blues
    bsk_colors = ["#a1d99b", "#31a354", "#006d2c"]  # greens
    comp_labels = ["x", "y", "z"]

    # Get the chief satellite name
    skf_chief_sat_name = skf_list[0].satellite_name
    bsk_chief_sat_name = bsk_list[0].satellite_name
    if skf_chief_sat_name != bsk_chief_sat_name:
        raise ValueError(f"Mismatch between the first satellite (chief) name in skf_list ({skf_chief_sat_name} "
                         f"and bsk_list ({bsk_chief_sat_name})")
    chief_sat_name = skf_chief_sat_name

    for i in range(1, n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape sanity checks (won't fail if shapes are as specified)
        if skf.rel_pos.shape[0] != 3 or bsk.rel_pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D
        t_skf = np.ravel(skf.time) / (60*60) # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60*60) # Time [hours]

        # Create a new figure for this satellite; no explicit numbering to avoid conflicts
        plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax = plt.gca()

        # Plot SKF (blue shades)
        for comp in range(3):
            ax.plot(
                t_skf,
                skf.rel_pos[comp, :],
                label=f"SKF {comp_labels[comp]}",
                linewidth=1.8,
                color=skf_colors[comp],
            )

        # Plot BSK (green shades)
        for comp in range(3):
            ax.plot(
                t_bsk,
                bsk.rel_pos[comp, :],
                label=f"BSK {comp_labels[comp]}",
                linewidth=1.8,
                linestyle="--",
                color=bsk_colors[comp],
            )

        sat_name = skf.satellite_name if getattr(skf, "satellite_name", None) else f"Satellite {i+1}"
        ax.set_title(f"Relative position between {sat_name} (follower) and {chief_sat_name} (chief)")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("RTN Δposition (m)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2)

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)
    

def plot_simulator_state_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    For each satellite i, create a figure with two stacked subplots:
      Top:  (bsk.pos - skf.pos) components over time
      Bottom: (bsk.vel - skf.vel) components over time
    BSK data are interpolated onto the SKF time grid if their time vectors differ.
    Uses colors: x=red, y=green, z=blue. Does not call plt.show().
    """
    main_plt_identifier = "SimDiff" # Part of the saved figure's name

    # Colors for components: x (red), y (green), z (blue)
    COMP_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # r, g, b
    COMP_LABELS = ["x", "y", "z"]

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data
    data_processor = DataProcessor()

    if len(skf_list) != len(bsk_list):
        raise ValueError(f"Satellite count mismatch: SKF={len(skf_list)}, BSK={len(bsk_list)}")

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"[sat {i}] pos must be shape (3, n)")
        if skf.vel.shape[0] != 3 or bsk.vel.shape[0] != 3:
            raise ValueError(f"[sat {i}] vel must be shape (3, n)")

        # Flatten/validate times and ensure increasing for interpolation
        t_skf = np.ravel(skf.time) / (60*60) # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60*60) # Time [hours]
        if t_skf.size != skf.pos.shape[1] or t_skf.size != skf.vel.shape[1]:
            raise ValueError(f"[sat {i}] SKF time length must match pos/vel columns")
        if t_bsk.size != bsk.pos.shape[1] or t_bsk.size != bsk.vel.shape[1]:
            raise ValueError(f"[sat {i}] BSK time length must match pos/vel columns")

        t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf.pos)
        _,     skf_vel = data_processor.ensure_increasing(t_skf, skf.vel)  # skf_pos/vel share t_skf order

        t_bsk, bsk_pos = data_processor.ensure_increasing(t_bsk, bsk.pos)
        _,     bsk_vel = data_processor.ensure_increasing(t_bsk, bsk.vel)

        # Interpolate BSK onto SKF time grid if needed
        if t_bsk.size != t_skf.size or not np.allclose(t_bsk, t_skf):
            bsk_pos_on_skf = data_processor.interp_3xn(t_bsk, bsk_pos, t_skf)
            bsk_vel_on_skf = data_processor.interp_3xn(t_bsk, bsk_vel, t_skf)
        else:
            bsk_pos_on_skf = bsk_pos
            bsk_vel_on_skf = bsk_vel

        # Differences
        dpos = bsk_pos_on_skf - skf_pos
        dvel = bsk_vel_on_skf - skf_vel

        # Create figure with two stacked subplots; no explicit numbering to avoid conflicts
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(PLT_WIDTH, PLT_HEIGHT))
        ax_pos, ax_vel = axes

        # Top: position diffs
        for comp in range(3):
            ax_pos.plot(
                t_skf, dpos[comp],
                label=f"Δpos {COMP_LABELS[comp]}",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_pos.set_ylabel("ECI Δposition (m)")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(ncol=3)

        # Bottom: velocity diffs
        for comp in range(3):
            ax_vel.plot(
                t_skf, dvel[comp],
                label=f"Δvel {COMP_LABELS[comp]}",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_vel.set_xlabel("Time (hours)")
        ax_vel.set_ylabel("ECI Δvelocity (m/s)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.legend(ncol=3)

        sat_name = getattr(skf, "satellite_name", None) or f"Satellite {i+1}"
        fig.suptitle(f"Simulator difference — {sat_name}")
        # fig.tight_layout(rect=[0., 0., 1., 0.96])
        
        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_simulator_rel_state_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    # TODO: Fix chat language
    """
    Plot simulator differences between relative RTN vectors (BSK - SKF).

    Assumes `DataProcessor.calculate_relative_formation_movement_rtc(...)` has already
    been called on both `skf_sim_data` and `bsk_sim_data`, so that `rel_data`
    is populated for each.

    For each follower satellite i (i = 1..n_sats-1), creates a figure with two stacked subplots:
      Top:    (bsk.rel_pos - skf.rel_pos) in RTN, components over time
      Bottom: (bsk.rel_vel - skf.rel_vel) in RTN, components over time

    Uses colors: x=red, y=green, z=blue.
    """
    main_plt_identifier = "SimRelDiff"  # Part of the saved figure's name

    # Colors for components: x (red), y (green), z (blue)
    COMP_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # r, g, b
    COMP_LABELS = ["x", "y", "z"]

    data_processor = DataProcessor()

    skf_list = skf_sim_data.rel_data
    bsk_list = bsk_sim_data.rel_data

    # Ensure relative data has been computed
    if skf_list is None or bsk_list is None:
        raise ValueError(
            "Relative RTN data not found in skf_sim_data/bsk_sim_data. "
            "Make sure plot_rel_pos_comparison (or the underlying "
            "calculate_relative_formation_movement_rtc) has been run first."
        )

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch in rel_data: SKF={len(skf_list)}, BSK={len(bsk_list)}"
        )

    n_sats = len(skf_list)
    if n_sats <= 1:
        # Need at least a chief + one follower
        return

    # Chief is assumed to be index 0 in rel_data (as in plot_rel_pos_comparison)
    skf_chief_sat_name = skf_list[0].satellite_name
    bsk_chief_sat_name = bsk_list[0].satellite_name
    if skf_chief_sat_name != bsk_chief_sat_name:
        raise ValueError(
            "Mismatch between chief satellite names in rel_data: "
            f"SKF chief={skf_chief_sat_name}, BSK chief={bsk_chief_sat_name}"
        )
    chief_sat_name = skf_chief_sat_name

    # Loop over followers only (1..n_sats-1)
    for i in range(1, n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape checks for relative pos/vel
        if skf.rel_pos.shape[0] != 3 or bsk.rel_pos.shape[0] != 3:
            raise ValueError(f"[rel sat {i}] rel_pos must be shape (3, n)")
        if not hasattr(skf, "rel_vel") or not hasattr(bsk, "rel_vel"):
            raise ValueError(
                f"[rel sat {i}] rel_vel attribute missing on relative data objects. "
                "Ensure calculate_relative_formation_movement_rtc computes rel_vel."
            )
        if skf.rel_vel.shape[0] != 3 or bsk.rel_vel.shape[0] != 3:
            raise ValueError(f"[rel sat {i}] rel_vel must be shape (3, n)")

        # Flatten/validate times and ensure increasing for interpolation
        t_skf = np.ravel(skf.time) / (60 * 60)  # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60 * 60)  # Time [hours]

        if t_skf.size != skf.rel_pos.shape[1] or t_skf.size != skf.rel_vel.shape[1]:
            raise ValueError(
                f"[rel sat {i}] SKF time length must match rel_pos/rel_vel columns"
            )
        if t_bsk.size != bsk.rel_pos.shape[1] or t_bsk.size != bsk.rel_vel.shape[1]:
            raise ValueError(
                f"[rel sat {i}] BSK time length must match rel_pos/rel_vel columns"
            )

        # Ensure increasing time and aligned ordering
        t_skf, skf_rel_pos = data_processor.ensure_increasing(t_skf, skf.rel_pos)
        _,     skf_rel_vel = data_processor.ensure_increasing(t_skf, skf.rel_vel)

        t_bsk, bsk_rel_pos = data_processor.ensure_increasing(t_bsk, bsk.rel_pos)
        _,     bsk_rel_vel = data_processor.ensure_increasing(t_bsk, bsk.rel_vel)

        # Interpolate BSK relative data onto SKF time grid if needed
        if t_bsk.size != t_skf.size or not np.allclose(t_bsk, t_skf):
            bsk_rel_pos_on_skf = data_processor.interp_3xn(t_bsk, bsk_rel_pos, t_skf)
            bsk_rel_vel_on_skf = data_processor.interp_3xn(t_bsk, bsk_rel_vel, t_skf)
        else:
            bsk_rel_pos_on_skf = bsk_rel_pos
            bsk_rel_vel_on_skf = bsk_rel_vel

        # Differences (BSK - SKF) in RTN
        d_rel_pos = bsk_rel_pos_on_skf - skf_rel_pos
        d_rel_vel = bsk_rel_vel_on_skf - skf_rel_vel

        # Create figure with two stacked subplots; no explicit numbering
        fig, axes = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(PLT_WIDTH, PLT_HEIGHT)
        )
        ax_pos, ax_vel = axes

        # Top: relative position diffs (RTN)
        for comp in range(3):
            ax_pos.plot(
                t_skf,
                d_rel_pos[comp],
                label=f"Δrel_pos {COMP_LABELS[comp]} (RTN)",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_pos.set_ylabel("RTN Δrel position (m)")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(ncol=3)

        # Bottom: relative velocity diffs (RTN)
        for comp in range(3):
            ax_vel.plot(
                t_skf,
                d_rel_vel[comp],
                label=f"Δrel_vel {COMP_LABELS[comp]} (RTN)",
                linewidth=1.8,
                color=COMP_COLORS[comp],
            )
        ax_vel.set_xlabel("Time (hours)")
        ax_vel.set_ylabel("RTN Δrel velocity (m/s)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.legend(ncol=3)

        sat_name = getattr(skf, "satellite_name", None) or f"Satellite {i}"
        fig.suptitle(
            f"Simulator RTN relative difference — {sat_name} (follower) vs {chief_sat_name} (chief)"
        )

        # Save via your existing helper
        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}_vs_{chief_sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_altitude_comparison(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    Plot altitude vs time for each satellite, for both Skyfield and Basilisk, in a single figure.

    Altitude is defined as ||pos|| - R_earth_mean, where pos is the ECI position vector.
    """
    main_plt_identifier = "AltComp"  # Used in the saved plot name
    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF has {len(skf_list)}, BSK has {len(bsk_list)}."
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    # Mean Earth radius [m] (approx. WGS-84 mean radius)
    EARTH_MEAN_RADIUS_M = 6371e3

    # Choose some colors for different satellites (will cycle if more sats than colors)
    sat_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]

    # Create a single figure for all satellites
    plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
    ax = plt.gca()

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape sanity checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"pos must be shape (3, n) for satellite index {i}.")
        if skf.time.ndim not in (1, 2) or bsk.time.ndim not in (1, 2):
            raise ValueError(f"time must be 1D or (1, n) for satellite index {i}.")

        # Flatten times to 1D and convert to hours
        t_skf = np.ravel(skf.time) / (60 * 60)  # [hours]
        t_bsk = np.ravel(bsk.time) / (60 * 60)  # [hours]

        # Compute altitude = ||pos|| - R_earth_mean
        skf_r = np.linalg.norm(skf.pos, axis=0)
        bsk_r = np.linalg.norm(bsk.pos, axis=0)
        skf_alt = skf_r - EARTH_MEAN_RADIUS_M
        bsk_alt = bsk_r - EARTH_MEAN_RADIUS_M

        color = sat_colors[i % len(sat_colors)]
        sat_name = (
            skf.satellite_name
            if getattr(skf, "satellite_name", None)
            else f"Satellite {i+1}"
        )

        # Skyfield: solid line
        ax.plot(
            t_skf,
            skf_alt,
            label=f"SKF {sat_name}",
            linewidth=1.8,
            linestyle="-",
            color=color,
        )

        # Basilisk: dashed line, same color
        ax.plot(
            t_bsk,
            bsk_alt,
            label=f"BSK {sat_name}",
            linewidth=1.8,
            linestyle="--",
            color=color,
        )

    ax.set_title("Altitude comparison for all satellites")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    fig = plt.gcf()
    plt_identifier = f"{main_plt_identifier}_AllSats"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)


def plot_simulator_state_abs_diff_old(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    For each satellite i, create a figure with two stacked subplots:
      Top:    absolute difference in position magnitude  | |r_BSK| - |r_SKF| |
      Bottom: absolute difference in velocity magnitude | |v_BSK| - |v_SKF| |
    BSK data are interpolated onto the SKF time grid if their time vectors differ.
    """
    main_plt_identifier = "SimAbsDiff"  # Part of the saved figure's name

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data
    data_processor = DataProcessor()

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF={len(skf_list)}, BSK={len(bsk_list)}"
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"[sat {i}] pos must be shape (3, n)")
        if skf.vel.shape[0] != 3 or bsk.vel.shape[0] != 3:
            raise ValueError(f"[sat {i}] vel must be shape (3, n)")

        # Flatten/validate times and ensure increasing for interpolation
        t_skf = np.ravel(skf.time) / (60 * 60)  # Time [hours]
        t_bsk = np.ravel(bsk.time) / (60 * 60)  # Time [hours]
        if t_skf.size != skf.pos.shape[1] or t_skf.size != skf.vel.shape[1]:
            raise ValueError(
                f"[sat {i}] SKF time length must match pos/vel columns"
            )
        if t_bsk.size != bsk.pos.shape[1] or t_bsk.size != bsk.vel.shape[1]:
            raise ValueError(
                f"[sat {i}] BSK time length must match pos/vel columns"
            )

        t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf.pos)
        _,     skf_vel = data_processor.ensure_increasing(t_skf, skf.vel)

        t_bsk, bsk_pos = data_processor.ensure_increasing(t_bsk, bsk.pos)
        _,     bsk_vel = data_processor.ensure_increasing(t_bsk, bsk.vel)

        # Interpolate BSK onto SKF time grid if needed
        if t_bsk.size != t_skf.size or not np.allclose(t_bsk, t_skf):
            bsk_pos_on_skf = data_processor.interp_3xn(t_bsk, bsk_pos, t_skf)
            bsk_vel_on_skf = data_processor.interp_3xn(t_bsk, bsk_vel, t_skf)
        else:
            bsk_pos_on_skf = bsk_pos
            bsk_vel_on_skf = bsk_vel

        # Magnitudes
        skf_pos_norm = np.linalg.norm(skf_pos, axis=0)
        bsk_pos_norm = np.linalg.norm(bsk_pos_on_skf, axis=0)
        skf_vel_norm = np.linalg.norm(skf_vel, axis=0)
        bsk_vel_norm = np.linalg.norm(bsk_vel_on_skf, axis=0)

        # Absolute scalar differences
        dpos = bsk_pos_norm - skf_pos_norm
        dvel = bsk_vel_norm - skf_vel_norm

        # Create figure with two stacked subplots; no explicit numbering to avoid conflicts
        fig, axes = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(PLT_WIDTH, PLT_HEIGHT)
        )
        ax_pos, ax_vel = axes

        # Top: absolute position magnitude diff
        ax_pos.plot(
            t_skf,
            dpos,
            label="|r_BSK| - |r_SKF|",
            linewidth=1.8,
            color="#d62728",  # red
        )
        ax_pos.set_ylabel("Δ|r| (m)")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(ncol=1)

        # Bottom: absolute velocity magnitude diff
        ax_vel.plot(
            t_skf,
            dvel,
            label="|v_BSK| - |v_SKF|",
            linewidth=1.8,
            color="#1f77b4",  # blue
        )
        ax_vel.set_xlabel("Time (hours)")
        ax_vel.set_ylabel("Δ|v| (m/s)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.legend(ncol=1)

        sat_name = getattr(skf, "satellite_name", None) or f"Satellite {i+1}"
        fig.suptitle(f"Simulator ECI state difference — {sat_name}")

        fig = plt.gcf()
        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)


def plot_simulator_state_abs_diff(cfg: Config, skf_sim_data: SimData, bsk_sim_data: SimData) -> None:
    """
    For each satellite i, create a figure with two stacked subplots:
      Top:    difference in position magnitude   (|r_BSK| - |r_SKF|)
      Bottom: difference in velocity magnitude   (|v_BSK| - |v_SKF|)
    Visualization matches the "baseline vs others" style:
      - Baseline (SKF) is a horizontal zero line in each subplot
      - Basilisk is plotted as difference relative to the baseline
      - Single legend box for whole figure (top-right over top subplot)
    BSK data are interpolated onto the SKF time grid if their time vectors differ.
    """
    main_plt_identifier = "SimAbsDiff"

    BASE_LABEL = "Skf: SGP4 (base)"
    BSK_LABEL = "Bsk: RKF78, All"

    skf_list = skf_sim_data.sim_data
    bsk_list = bsk_sim_data.sim_data
    data_processor = DataProcessor()

    if len(skf_list) != len(bsk_list):
        raise ValueError(
            f"Satellite count mismatch: SKF={len(skf_list)}, BSK={len(bsk_list)}"
        )

    n_sats = len(skf_list)
    if n_sats == 0:
        return

    for i in range(n_sats):
        skf = skf_list[i]
        bsk = bsk_list[i]

        # Basic shape checks
        if skf.pos.shape[0] != 3 or bsk.pos.shape[0] != 3:
            raise ValueError(f"[sat {i}] pos must be shape (3, n)")
        if skf.vel.shape[0] != 3 or bsk.vel.shape[0] != 3:
            raise ValueError(f"[sat {i}] vel must be shape (3, n)")

        # Flatten/validate times and ensure increasing for interpolation
        t_skf = np.ravel(skf.time) / 3600.0  # [hours]
        t_bsk = np.ravel(bsk.time) / 3600.0  # [hours]
        if t_skf.size != skf.pos.shape[1] or t_skf.size != skf.vel.shape[1]:
            raise ValueError(f"[sat {i}] SKF time length must match pos/vel columns")
        if t_bsk.size != bsk.pos.shape[1] or t_bsk.size != bsk.vel.shape[1]:
            raise ValueError(f"[sat {i}] BSK time length must match pos/vel columns")

        t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf.pos)
        _,     skf_vel = data_processor.ensure_increasing(t_skf, skf.vel)

        t_bsk, bsk_pos = data_processor.ensure_increasing(t_bsk, bsk.pos)
        _,     bsk_vel = data_processor.ensure_increasing(t_bsk, bsk.vel)

        # Interpolate BSK onto SKF time grid if needed
        if t_bsk.size != t_skf.size or not np.allclose(t_bsk, t_skf):
            bsk_pos_on_skf = data_processor.interp_3xn(t_bsk, bsk_pos, t_skf)
            bsk_vel_on_skf = data_processor.interp_3xn(t_bsk, bsk_vel, t_skf)
        else:
            bsk_pos_on_skf = bsk_pos
            bsk_vel_on_skf = bsk_vel

        # Magnitudes
        skf_pos_norm = np.linalg.norm(skf_pos, axis=0)
        bsk_pos_norm = np.linalg.norm(bsk_pos_on_skf, axis=0)
        skf_vel_norm = np.linalg.norm(skf_vel, axis=0)
        bsk_vel_norm = np.linalg.norm(bsk_vel_on_skf, axis=0)

        # Differences (BSK - SKF)
        dpos = bsk_pos_norm - skf_pos_norm
        dvel = bsk_vel_norm - skf_vel_norm

        # Create figure
        fig, (ax_pos, ax_vel) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(PLT_WIDTH, PLT_HEIGHT)
        )

        # Baseline: horizontal zero line (shown in BOTH subplots)
        ax_pos.plot(t_skf, np.zeros_like(t_skf), label=BASE_LABEL, linewidth=1.8)
        ax_vel.plot(t_skf, np.zeros_like(t_skf), label=BASE_LABEL, linewidth=1.8)

        # Basilisk differences
        ax_pos.plot(
            t_skf,
            dpos,
            label=f"{BSK_LABEL}  (|r_BSK| - |r_SKF|)",
            linewidth=1.8,
        )
        ax_vel.plot(
            t_skf,
            dvel,
            label=f"{BSK_LABEL}  (|v_BSK| - |v_SKF|)",
            linewidth=1.8,
        )

        ax_pos.set_ylabel("Δ|r| (m)")
        ax_pos.grid(True, alpha=0.3)
        ax_pos.set_title("Simulator Difference in Position Magnitude")

        ax_vel.set_xlabel("Time (hours)")
        ax_vel.set_ylabel("Δ|v| (m/s)")
        ax_vel.grid(True, alpha=0.3)
        ax_vel.set_title("Simulator Difference in Velocity Magnitude")

        sat_name = getattr(skf, "satellite_name", None) or f"Satellite {i+1}"
        fig.suptitle(f"Simulator absolute state difference — {sat_name}")

        # Single shared legend (top-right over top subplot)
        handles_pos, labels_pos = ax_pos.get_legend_handles_labels()
        handles_vel, labels_vel = ax_vel.get_legend_handles_labels()

        seen = set()
        handles_all, labels_all = [], []
        for h, lab in zip(handles_pos + handles_vel, labels_pos + labels_vel):
            if lab in seen:
                continue
            seen.add(lab)
            handles_all.append(h)
            labels_all.append(lab)

        ax_pos.legend(
            handles_all,
            labels_all,
            loc="upper right",
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            ncol=1,
        )

        plt_identifier = f"{main_plt_identifier}_{sat_name}"
        conditional_save_plot(cfg, fig, plt_identifier)
        conditional_show_plot(cfg)




def plot_rel_pos_multi_sim_diff(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in the chief-follower RTN relative frame.

    Output: ONE figure with 3 stacked subplots:
      Top:    simulator difference in along-track (T) direction
      Middle: simulator difference in cross-track (N) direction
      Bottom: simulator difference in radial (R) direction

    bsk_data_entries : list of tuples
        Each entry is: (timestamp_string, legend_label)
        Example: [("20250101_120500", "Basilisk #1"), ...]
    """
    main_plt_identifier = "RelPosMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # --------- Load Skyfield baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_skf" # BE VEEEEEEERY CAREFUL TO CHECK THIS !!!!!
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No Skyfield datafiles found for timestamp '{skf_stamp}'")
    skf_filename = skf_files[0]
    skf_sim_data = data_loader.load_sim_data_file(skf_filename)

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)

    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")

        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Compute RTN relative positions (with downsampling) ---------
    data_processor.calculate_relative_formation_movement_rtc(skf_sim_data, RTN_DOWNSAMP_FAC)
    for (_, bsk_sim_data, _) in bsk_sim_data_list:
        data_processor.calculate_relative_formation_movement_rtc(bsk_sim_data, RTN_DOWNSAMP_FAC)

    skf_rel_list = skf_sim_data.rel_data
    if skf_rel_list is None:
        raise ValueError("Skyfield rel_data not computed correctly.")

    follower_index = 1  # Chief is index 0, follower at index 1

    skf_rel_follower = skf_rel_list[follower_index]
    chief_sat_name = skf_rel_list[0].satellite_name
    follower_name = skf_rel_follower.satellite_name

    # Baseline time grid (hours) and ensure increasing
    t_skf = np.ravel(skf_rel_follower.time) / (60 * 60)
    t_skf, skf_rel_pos = data_processor.ensure_increasing(t_skf, skf_rel_follower.rel_pos)

    # --------- Assign Colors ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, rel_data_object, color)
    datasets: list[tuple[str, RelObjData, str]] = []

    # Skyfield baseline (first entry)
    skf_label = base_label
    datasets.append((skf_label, skf_rel_follower, get_color(0)))

    # Basilisk datasets
    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_rel_list = bsk_sim_data.rel_data
        if bsk_rel_list is None:
            raise ValueError(f"Basilisk rel_data failed for timestamp {ts}")

        rel_obj = bsk_rel_list[follower_index]
        datasets.append((legend_label, rel_obj, get_color(i)))

    # --------- Helper: compute component differences on SKF grid ---------
    def compute_component_diffs(comp_idx: int) -> dict[str, np.ndarray]:
        diffs: dict[str, np.ndarray] = {}
        for label, rel_obj, _ in datasets:
            if label == base_label:
                # Baseline: zero difference against itself
                diffs[label] = np.zeros_like(skf_rel_pos[comp_idx])
                continue

            t = np.ravel(rel_obj.time) / (60 * 60)
            t, rel_pos = data_processor.ensure_increasing(t, rel_obj.rel_pos)

            if t.size != t_skf.size or not np.allclose(t, t_skf):
                rel_pos_on_skf = data_processor.interp_3xn(t, rel_pos, t_skf)
            else:
                rel_pos_on_skf = rel_pos

            diffs[label] = rel_pos_on_skf[comp_idx] - skf_rel_pos[comp_idx]

        return diffs

    # RTN indices (R,T,N) = (0,1,2)
    IDX_R = 0
    IDX_T = 1
    IDX_N = 2

    # Compute diffs for each component
    diffs_T = compute_component_diffs(IDX_T)
    diffs_N = compute_component_diffs(IDX_N)
    diffs_R = compute_component_diffs(IDX_R)

    # --------- Create a single figure with 3 stacked subplots ---------
    fig, (ax_T, ax_N, ax_R) = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    # Top: Along-track (T)
    for label, _, color in datasets:
        ax_T.plot(t_skf, diffs_T[label], label=label, color=color, linewidth=1.5)
    ax_T.set_ylabel("RTN Δy_rel (m)")
    ax_T.set_title("Along-track (T) simulator difference (y_rel - y_rel_base)")
    ax_T.grid(True, alpha=0.3)
    # ax_T.legend(ncol=2)

    # Middle: Cross-track (N)
    for label, _, color in datasets:
        ax_N.plot(t_skf, diffs_N[label], label=label, color=color, linewidth=1.5)
    ax_N.set_ylabel("RTN Δz_rel (m)")
    ax_N.set_title("Cross-track (N) simulator difference (z_rel - z_rel_base)")
    ax_N.grid(True, alpha=0.3)
    # ax_N.legend(ncol=2)

    # Bottom: Radial (R)
    for label, _, color in datasets:
        ax_R.plot(t_skf, diffs_R[label], label=label, color=color, linewidth=1.5)
    ax_R.set_ylabel("RTN Δx_rel (m)")
    ax_R.set_xlabel("Time (hours)")
    ax_R.set_title("Radial (R) simulator difference (x_rel - x_rel_base)")
    ax_R.grid(True, alpha=0.3)
    # ax_R.legend(ncol=2)

    # ---- Single shared legend for all datasets ----
    handles, labels = ax_T.get_legend_handles_labels()

    ax_T.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"Simulator Comparison: Leader-Follower RTN Relative Position\n"
        f"Leader: {chief_sat_name}, Follower: {follower_name}"
    )

    plt_identifier = f"{main_plt_identifier}_Diff_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)


def plot_rel_pos_multi_sim_diff_no_radial(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in the chief-follower RTN relative frame.

    Output: ONE figure with 2 stacked subplots:
      Top:    simulator difference in along-track (T) direction
      Middle: simulator difference in cross-track (N) direction

    bsk_data_entries : list of tuples
        Each entry is: (timestamp_string, legend_label)
        Example: [("20250101_120500", "Basilisk #1"), ...]
    """
    main_plt_identifier = "RelPosMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # --------- Load Skyfield baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_bsk" # BE VEEEEEEERY CAREFUL TO CHECK THIS !!!!!
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No Skyfield datafiles found for timestamp '{skf_stamp}'")
    skf_filename = skf_files[0]
    skf_sim_data = data_loader.load_sim_data_file(skf_filename)

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)

    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")

        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Compute RTN relative positions (with downsampling) ---------
    data_processor.calculate_relative_formation_movement_rtc(skf_sim_data, RTN_DOWNSAMP_FAC)
    for (_, bsk_sim_data, _) in bsk_sim_data_list:
        data_processor.calculate_relative_formation_movement_rtc(bsk_sim_data, RTN_DOWNSAMP_FAC)

    skf_rel_list = skf_sim_data.rel_data
    if skf_rel_list is None:
        raise ValueError("Skyfield rel_data not computed correctly.")

    follower_index = 1  # Chief is index 0, follower at index 1

    skf_rel_follower = skf_rel_list[follower_index]
    chief_sat_name = skf_rel_list[0].satellite_name
    follower_name = skf_rel_follower.satellite_name

    # Baseline time grid (hours) and ensure increasing
    t_skf = np.ravel(skf_rel_follower.time) / (60 * 60)
    t_skf, skf_rel_pos = data_processor.ensure_increasing(t_skf, skf_rel_follower.rel_pos)

    # --------- Assign Colors ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, rel_data_object, color)
    datasets: list[tuple[str, RelObjData, str]] = []

    # Skyfield baseline (first entry)
    skf_label = base_label
    datasets.append((skf_label, skf_rel_follower, get_color(0)))

    # Basilisk datasets
    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_rel_list = bsk_sim_data.rel_data
        if bsk_rel_list is None:
            raise ValueError(f"Basilisk rel_data failed for timestamp {ts}")

        rel_obj = bsk_rel_list[follower_index]
        datasets.append((legend_label, rel_obj, get_color(i)))

    # --------- Helper: compute component differences on SKF grid ---------
    def compute_component_diffs(comp_idx: int) -> dict[str, np.ndarray]:
        diffs: dict[str, np.ndarray] = {}
        for label, rel_obj, _ in datasets:
            if label == base_label:
                # Baseline: zero difference against itself
                diffs[label] = np.zeros_like(skf_rel_pos[comp_idx])
                continue

            t = np.ravel(rel_obj.time) / (60 * 60)
            t, rel_pos = data_processor.ensure_increasing(t, rel_obj.rel_pos)

            if t.size != t_skf.size or not np.allclose(t, t_skf):
                rel_pos_on_skf = data_processor.interp_3xn(t, rel_pos, t_skf)
            else:
                rel_pos_on_skf = rel_pos

            diffs[label] = rel_pos_on_skf[comp_idx] - skf_rel_pos[comp_idx]

        return diffs

    # RTN indices (R,T,N) = (0,1,2)
    IDX_R = 0
    IDX_T = 1
    IDX_N = 2

    # Compute diffs for each component
    diffs_T = compute_component_diffs(IDX_T)
    diffs_N = compute_component_diffs(IDX_N)
    diffs_R = compute_component_diffs(IDX_R)

    # --------- Create a single figure with 2 stacked subplots ---------
    fig, (ax_T, ax_N) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    # Top: Along-track (T)
    for label, _, color in datasets:
        ax_T.plot(t_skf, diffs_T[label], label=label, color=color, linewidth=1.5)
    ax_T.set_ylabel("RTN Δy_rel (m)")
    ax_T.set_title("Along-track (T) simulator difference (y_rel - y_rel_base)")
    ax_T.grid(True, alpha=0.3)
    # ax_T.legend(ncol=2)

    # Middle: Cross-track (N)
    for label, _, color in datasets:
        ax_N.plot(t_skf, diffs_N[label], label=label, color=color, linewidth=1.5)
    ax_N.set_ylabel("RTN Δz_rel (m)")
    ax_N.set_xlabel("Time (hours)")
    ax_N.set_title("Cross-track (N) simulator difference (z_rel - z_rel_base)")
    ax_N.grid(True, alpha=0.3)
    # ax_N.legend(ncol=2)

    # ---- Single shared legend for all datasets ----
    handles, labels = ax_T.get_legend_handles_labels()

    ax_T.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"Simulator Comparison: Leader-Follower RTN Relative Position\n"
        f"Leader: {chief_sat_name}, Follower: {follower_name}"
    )

    plt_identifier = f"{main_plt_identifier}_Diff_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)



def plot_rel_vel_multi_sim_diff(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in the chief-follower RTN relative-velocity frame.

    Output: ONE figure with 3 stacked subplots:
      Top:    simulator difference in along-track (T) relative velocity
      Middle: simulator difference in cross-track (N) relative velocity
      Bottom: simulator difference in radial (R) relative velocity

    bsk_data_entries : list of tuples
        Each entry is: (timestamp_string, legend_label)
        Example: [("20250101_120500", "Basilisk #1"), ...]
    """
    main_plt_identifier = "RelVelMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # --------- Load Skyfield baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_bsk" # REMEMBER TO SELECT CORRECT STRING ENDING
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No Skyfield datafiles found for timestamp '{skf_stamp}'")
    skf_filename = skf_files[0]
    skf_sim_data = data_loader.load_sim_data_file(skf_filename)

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)

    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")

        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Compute RTN relative positions/velocities (with downsampling) ---------
    data_processor.calculate_relative_formation_movement_rtc(skf_sim_data, RTN_DOWNSAMP_FAC)
    for (_, bsk_sim_data, _) in bsk_sim_data_list:
        data_processor.calculate_relative_formation_movement_rtc(bsk_sim_data, RTN_DOWNSAMP_FAC)

    skf_rel_list = skf_sim_data.rel_data
    if skf_rel_list is None:
        raise ValueError("Skyfield rel_data not computed correctly.")

    follower_index = 1  # Chief is index 0, follower at index 1

    skf_rel_follower = skf_rel_list[follower_index]
    chief_sat_name = skf_rel_list[0].satellite_name
    follower_name = skf_rel_follower.satellite_name

    # Baseline time grid (hours) and ensure increasing
    t_skf = np.ravel(skf_rel_follower.time) / (60 * 60)
    t_skf, skf_rel_vel = data_processor.ensure_increasing(t_skf, skf_rel_follower.rel_vel)

    # --------- Assign Colors ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, rel_data_object, color)
    datasets: list[tuple[str, RelObjData, str]] = []

    # Skyfield baseline (first entry)
    skf_label = base_label
    datasets.append((skf_label, skf_rel_follower, get_color(0)))

    # Basilisk datasets
    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_rel_list = bsk_sim_data.rel_data
        if bsk_rel_list is None:
            raise ValueError(f"Basilisk rel_data failed for timestamp {ts}")

        rel_obj = bsk_rel_list[follower_index]
        datasets.append((legend_label, rel_obj, get_color(i)))

    # --------- Helper: compute component differences on SKF grid (velocities) ---------
    def compute_component_diffs(comp_idx: int) -> dict[str, np.ndarray]:
        diffs: dict[str, np.ndarray] = {}
        for label, rel_obj, _ in datasets:
            if label == base_label:
                # Baseline: zero difference against itself
                diffs[label] = np.zeros_like(skf_rel_vel[comp_idx])
                continue

            t = np.ravel(rel_obj.time) / (60 * 60)
            t, rel_vel = data_processor.ensure_increasing(t, rel_obj.rel_vel)

            if t.size != t_skf.size or not np.allclose(t, t_skf):
                rel_vel_on_skf = data_processor.interp_3xn(t, rel_vel, t_skf)
            else:
                rel_vel_on_skf = rel_vel

            diffs[label] = rel_vel_on_skf[comp_idx] - skf_rel_vel[comp_idx]

        return diffs

    # RTN indices (R,T,N) = (0,1,2)
    IDX_R = 0
    IDX_T = 1
    IDX_N = 2

    # Compute diffs for each component
    diffs_T = compute_component_diffs(IDX_T)
    diffs_N = compute_component_diffs(IDX_N)
    diffs_R = compute_component_diffs(IDX_R)

    # --------- Create a single figure with 3 stacked subplots ---------
    fig, (ax_T, ax_N, ax_R) = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    # Top: Along-track (T) velocity diff
    for label, _, color in datasets:
        ax_T.plot(t_skf, diffs_T[label], label=label, color=color, linewidth=1.5)
    ax_T.set_ylabel("RTN Δv_rel (m/s)")
    ax_T.set_title("Along-track (T) simulator velocity difference (v_rel - v_rel_base)")
    ax_T.grid(True, alpha=0.3)

    # Middle: Cross-track (N) velocity diff
    for label, _, color in datasets:
        ax_N.plot(t_skf, diffs_N[label], label=label, color=color, linewidth=1.5)
    ax_N.set_ylabel("RTN Δw_rel (m/s)")
    ax_N.set_title("Cross-track (N) simulator velocity difference (w_rel - w_rel_base)")
    ax_N.grid(True, alpha=0.3)

    # Bottom: Radial (R) velocity diff
    for label, _, color in datasets:
        ax_R.plot(t_skf, diffs_R[label], label=label, color=color, linewidth=1.5)
    ax_R.set_ylabel("RTN Δu_rel (m/s)")
    ax_R.set_xlabel("Time (hours)")
    ax_R.set_title("Radial (R) simulator velocity difference (u_rel - u_rel_base)")
    ax_R.grid(True, alpha=0.3)

    # ---- Single shared legend in top-right of top subplot ----
    handles, labels = ax_T.get_legend_handles_labels()
    ax_T.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"Simulator Comparison: Leader-Follower RTN Relative Velocity\n"
        f"Leader: {chief_sat_name}, Follower: {follower_name}"
    )

    plt_identifier = f"{main_plt_identifier}_Diff_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)





def plot_alt_multi_sim_diff_old(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in absolute altitude for a given follower spacecraft.

    Output: ONE figure with 2 stacked subplots:
      Top:    altitude time series for all simulations
      Bottom: simulator difference in altitude relative to the Skyfield baseline

    Inputs:
        skf_base_data_timestamp : str
            Baseline timestamp (without suffix); '_skf' is appended internally.
        bsk_data_entries : list[tuple[str, str]]
            Each entry: (timestamp_string, legend_label) for a Basilisk run.
        base_label : str
            Legend label to use for the Skyfield baseline.

    Notes:
        - Uses the same follower index convention as plot_rel_pos_multi_sim_diff
          (chief at index 0, follower at index 1).
    """
    main_plt_identifier = "AltMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # ---- Constants ----
    # Mean Earth radius [m] (you can replace this with a project-wide constant if you have one)
    EARTH_MEAN_RADIUS_M = 6378137.0

    # --------- Load Skyfield baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_skf"
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No Skyfield datafiles found for timestamp '{skf_stamp}'")
    skf_filename = skf_files[0]
    skf_sim_data = data_loader.load_sim_data_file(skf_filename)

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)

    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")

        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Select chief and follower spacecraft ---------
    skf_obj_list = skf_sim_data.sim_data
    if len(skf_obj_list) <= 1:
        raise ValueError("Need at least a chief and one follower in the Skyfield dataset.")

    chief_obj = skf_obj_list[0]
    chief_index = 0
    skf_chief = skf_obj_list[chief_index]

    chief_sat_name = chief_obj.satellite_name
    follower_name = skf_chief.satellite_name

    # --------- Baseline time grid & altitude ---------
    # Time in hours
    t_skf = np.ravel(skf_chief.time) / (60 * 60)
    # Ensure increasing, applied to position (3 x n)
    t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf_chief.pos)

    # Altitude = |r| - R_earth [km]
    skf_alt = (np.linalg.norm(skf_pos, axis=0) - EARTH_MEAN_RADIUS_M) / 1000

    # --------- Assign Colors ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, time_hours, altitude, color)
    datasets: list[tuple[str, np.ndarray, np.ndarray, str]] = []

    # Skyfield baseline (first entry)
    datasets.append((base_label, t_skf, skf_alt, get_color(0)))

    # Basilisk datasets
    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_obj_list = bsk_sim_data.sim_data
        if len(bsk_obj_list) <= chief_index:
            raise ValueError(f"Basilisk dataset for timestamp {ts} has no follower at index {chief_index}")

        bsk_chief = bsk_obj_list[chief_index]

        t = np.ravel(bsk_chief.time) / (60 * 60)
        t, bsk_pos = data_processor.ensure_increasing(t, bsk_chief.pos)
        alt = (np.linalg.norm(bsk_pos, axis=0) - EARTH_MEAN_RADIUS_M) / 1000 # [km]

        datasets.append((legend_label, t, alt, get_color(i)))

    # --------- Compute altitude differences vs baseline on baseline time grid ---------
    alt_diffs: dict[str, np.ndarray] = {}
    for label, t, alt, _ in datasets:
        if label == base_label:
            alt_diffs[label] = np.zeros_like(skf_alt)
        else:
            # Interpolate alt to baseline time grid t_skf
            # Assumes t is sorted (guaranteed by ensure_increasing)
            alt_on_skf = np.interp(t_skf, t, alt)
            alt_diffs[label] = (alt_on_skf - skf_alt) * 1000 # [m]

    # --------- Create figure with 2 stacked subplots ---------
    fig, (ax_alt, ax_diff) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    # Top: absolute altitude
    for label, t, alt, color in datasets:
        ax_alt.plot(t, alt, label=label, color=color, linewidth=1.5)

    ax_alt.set_ylabel("Altitude (km)")
    ax_alt.set_title("Leader-Follower Altitude")
    ax_alt.grid(True, alpha=0.3)

    # Bottom: altitude difference vs baseline
    for label, _, _, color in datasets:
        ax_diff.plot(t_skf, alt_diffs[label], label=label, color=color, linewidth=1.5)

    ax_diff.set_xlabel("Time (hours)")
    ax_diff.set_ylabel("Δ Altitude vs Baseline (m)")
    ax_diff.set_title("Simulator Difference in Altitude")
    ax_diff.grid(True, alpha=0.3)

    # ---- Single shared legend in top-right of top subplot ----
    handles, labels = ax_alt.get_legend_handles_labels()
    ax_alt.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"Altitude and Simulator Difference for the Leader Spacecraft\n"
        f"Leader: {chief_sat_name}"
    )

    plt_identifier = f"{main_plt_identifier}_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)


def plot_alt_multi_sim_diff(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in absolute altitude for the leader spacecraft.

    Output: ONE figure with a single plot:
      - simulator difference in altitude relative to the Skyfield baseline.

    Inputs:
        skf_base_data_timestamp : str
            Baseline timestamp (without suffix); '_skf' is appended internally.
        bsk_data_entries : list[tuple[str, str]]
            Each entry: (timestamp_string, legend_label) for a Basilisk run.
        base_label : str
            Legend label to use for the Skyfield baseline.
    """
    main_plt_identifier = "AltMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # ---- Constants ----
    # Mean Earth radius [m]
    EARTH_MEAN_RADIUS_M = 6378137.0

    # --------- Load Skyfield baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_skf"
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No Skyfield datafiles found for timestamp '{skf_stamp}'")
    skf_filename = skf_files[0]
    skf_sim_data = data_loader.load_sim_data_file(skf_filename)

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)

    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")

        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Select leader spacecraft (index 0) ---------
    skf_obj_list = skf_sim_data.sim_data
    if len(skf_obj_list) <= 1:
        raise ValueError("Need at least a leader and one follower in the Skyfield dataset.")

    chief_obj = skf_obj_list[0]
    chief_index = 0
    skf_chief = skf_obj_list[chief_index]

    chief_sat_name = chief_obj.satellite_name
    follower_name = skf_chief.satellite_name  # kept to match your identifier naming

    # --------- Baseline time grid & altitude ---------
    # Time in hours
    t_skf = np.ravel(skf_chief.time) / (60 * 60)
    # Ensure increasing, applied to position (3 x n)
    t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf_chief.pos)

    # Altitude = |r| - R_earth [km]
    skf_alt = (np.linalg.norm(skf_pos, axis=0) - EARTH_MEAN_RADIUS_M) / 1000.0  # [km]

    # --------- Assign Colors ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, time_hours, altitude_km, color)
    datasets: list[tuple[str, np.ndarray, np.ndarray, str]] = []

    # Skyfield baseline (first entry)
    datasets.append((base_label, t_skf, skf_alt, get_color(0)))

    # Basilisk datasets
    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_obj_list = bsk_sim_data.sim_data
        if len(bsk_obj_list) <= chief_index:
            raise ValueError(
                f"Basilisk dataset for timestamp {ts} has no leader at index {chief_index}"
            )

        bsk_chief = bsk_obj_list[chief_index]

        t = np.ravel(bsk_chief.time) / (60 * 60)
        t, bsk_pos = data_processor.ensure_increasing(t, bsk_chief.pos)
        alt = (np.linalg.norm(bsk_pos, axis=0) - EARTH_MEAN_RADIUS_M) / 1000.0  # [km]

        datasets.append((legend_label, t, alt, get_color(i)))

    # --------- Compute altitude differences vs baseline on baseline time grid ---------
    # Differences in meters [m]
    alt_diffs: dict[str, np.ndarray] = {}
    for label, t, alt, _ in datasets:
        if label == base_label:
            alt_diffs[label] = np.zeros_like(skf_alt * 1000.0)  # [m]
        else:
            # Interpolate alt to baseline time grid t_skf [km]
            alt_on_skf = np.interp(t_skf, t, alt)  # [km]
            alt_diffs[label] = (alt_on_skf - skf_alt) * 1000.0  # [m]

    # --------- Create figure with a SINGLE plot (only diffs) ---------
    fig, ax_diff = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    # Bottom: altitude difference vs baseline
    for label, _, _, color in datasets:
        ax_diff.plot(t_skf, alt_diffs[label], label=label, color=color, linewidth=1.5)

    ax_diff.set_xlabel("Time (hours)")
    ax_diff.set_ylabel("Δ Altitude vs Baseline (m)")
    ax_diff.set_title("Simulator Difference in Altitude")
    ax_diff.grid(True, alpha=0.3)

    # ---- Single shared legend in top-right of the plot ----
    handles, labels = ax_diff.get_legend_handles_labels()
    ax_diff.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"Altitude Simulator Difference for the Leader Spacecraft\n"
        f"Leader: {chief_sat_name}"
    )

    plt_identifier = f"{main_plt_identifier}_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)


def plot_simulator_state_mag_multi_sim_diff(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in ECI state magnitude differences for the leader spacecraft.

    Output: ONE figure with 2 stacked subplots:
      Top:    simulator difference in position magnitude   (|r|_sim - |r|_baseline)   [m]
      Bottom: simulator difference in velocity magnitude   (|v|_sim - |v|_baseline)   [m/s]

    Inputs:
        skf_base_data_timestamp : str
            Baseline timestamp (without suffix); '_skf' is appended internally.
        bsk_data_entries : list[tuple[str, str]]
            Each entry: (timestamp_string, legend_label) for a Basilisk run.
        base_label : str
            Legend label to use for the Skyfield baseline.

    Notes:
        - Uses leader spacecraft at index 0 (same convention as plot_alt_multi_sim_diff).
        - Basilisk series are interpolated onto the Skyfield baseline time grid.
        - Legend labels are exactly the provided base_label and tuple legend strings.
    """
    main_plt_identifier = "SimStateAbsMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # --------- Load Skyfield baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_skf" # REMEMBER TO SELECT THE CORRECT STRING ENDING HERE!
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No Skyfield datafiles found for timestamp '{skf_stamp}'")
    skf_filename = skf_files[0]
    skf_sim_data = data_loader.load_sim_data_file(skf_filename)

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)
    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")
        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Select leader spacecraft (index 0) ---------
    chief_index = 0
    skf_obj_list = skf_sim_data.sim_data
    if len(skf_obj_list) <= chief_index:
        raise ValueError("Skyfield dataset has no leader spacecraft at index 0.")

    skf_chief = skf_obj_list[chief_index]
    chief_sat_name = skf_chief.satellite_name
    follower_name = skf_chief.satellite_name  # kept to match your identifier naming pattern

    # --------- Baseline time grid & baseline magnitudes ---------
    t_skf = np.ravel(skf_chief.time) / 3600.0  # [hours]
    t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf_chief.pos)
    _,     skf_vel = data_processor.ensure_increasing(t_skf, skf_chief.vel)

    skf_pos_norm = np.linalg.norm(skf_pos, axis=0)  # [m]
    skf_vel_norm = np.linalg.norm(skf_vel, axis=0)  # [m/s]

    # --------- Assign Colors (same approach as plot_alt_multi_sim_diff) ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, time_hours, pos_norm, vel_norm, color)
    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = []

    # Skyfield baseline (first entry)
    datasets.append((base_label, t_skf, skf_pos_norm, skf_vel_norm, get_color(0)))

    # Basilisk datasets
    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_obj_list = bsk_sim_data.sim_data
        if len(bsk_obj_list) <= chief_index:
            raise ValueError(f"Basilisk dataset for timestamp {ts} has no leader at index {chief_index}")

        bsk_chief = bsk_obj_list[chief_index]

        t = np.ravel(bsk_chief.time) / 3600.0  # [hours]
        t, bsk_pos = data_processor.ensure_increasing(t, bsk_chief.pos)
        _, bsk_vel = data_processor.ensure_increasing(t, bsk_chief.vel)

        bsk_pos_norm = np.linalg.norm(bsk_pos, axis=0)  # [m]
        bsk_vel_norm = np.linalg.norm(bsk_vel, axis=0)  # [m/s]

        datasets.append((legend_label, t, bsk_pos_norm, bsk_vel_norm, get_color(i)))

    # --------- Compute differences vs baseline on baseline time grid ---------
    # Differences:
    #   pos: meters [m]
    #   vel: m/s [m/s]
    pos_diffs: dict[str, np.ndarray] = {}
    vel_diffs: dict[str, np.ndarray] = {}

    for label, t, pos_norm, vel_norm, _ in datasets:
        if label == base_label:
            pos_diffs[label] = np.zeros_like(skf_pos_norm)  # [m]
            vel_diffs[label] = np.zeros_like(skf_vel_norm)  # [m/s]
        else:
            pos_on_skf = np.interp(t_skf, t, pos_norm)      # [m]
            vel_on_skf = np.interp(t_skf, t, vel_norm)      # [m/s]
            pos_diffs[label] = pos_on_skf - skf_pos_norm    # [m]
            vel_diffs[label] = vel_on_skf - skf_vel_norm    # [m/s]

    # --------- Create figure with 2 stacked subplots (diffs only) ---------
    fig, (ax_posdiff, ax_veldiff) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    # Top: position magnitude difference vs baseline
    for label, _, _, _, color in datasets:
        ax_posdiff.plot(t_skf, pos_diffs[label], label=label, color=color, linewidth=1.5)

    ax_posdiff.set_ylabel("ECI Δ|r| (m)")
    ax_posdiff.set_title("Simulator Difference Position Magnitude (|r| - |r_base|)")
    ax_posdiff.grid(True, alpha=0.3)

    # Bottom: velocity magnitude difference vs baseline
    for label, _, _, _, color in datasets:
        ax_veldiff.plot(t_skf, vel_diffs[label], label=label, color=color, linewidth=1.5)

    ax_veldiff.set_xlabel("Time (hours)")
    ax_veldiff.set_ylabel("ECI Δ|v| (m/s)")
    ax_veldiff.set_title("Simulator Difference Velocity Magnitude (|v| - |v_base|)")
    ax_veldiff.grid(True, alpha=0.3)

    # ---- Single shared legend in top-right over the top subplot ----
    handles, labels = ax_posdiff.get_legend_handles_labels()
    ax_posdiff.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"ECI State Magnitude Simulator Difference for the Leader Spacecraft\n"
        f"Leader: {chief_sat_name}"
    )

    plt_identifier = f"{main_plt_identifier}_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)


def plot_multi_sim_pos_vel_sim_diff_mag(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in ECI state *delta magnitude* differences for the leader spacecraft.

    Output: ONE figure with 2 stacked subplots:
      Top:    simulator difference in position delta magnitude   ||r_sim - r_base||   [m]
      Bottom: simulator difference in velocity delta magnitude   ||v_sim - v_base||   [m/s]

    Notes:
        - Uses leader spacecraft at index 0.
        - Basilisk series are interpolated onto the baseline time grid.
        - Baseline is plotted as a horizontal zero line using base_label.
    """
    main_plt_identifier = "SimStateAbsMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # --------- Load baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_bsk"  # REMEMBER TO SELECT THE CORRECT STRING ENDING!!!!
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No baseline datafiles found for timestamp '{skf_stamp}'")
    skf_sim_data = data_loader.load_sim_data_file(skf_files[0])

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)
    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")
        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Select leader spacecraft (index 0) ---------
    chief_index = 0
    skf_obj_list = skf_sim_data.sim_data
    if len(skf_obj_list) <= chief_index:
        raise ValueError("Baseline dataset has no leader spacecraft at index 0.")

    skf_chief = skf_obj_list[chief_index]
    chief_sat_name = skf_chief.satellite_name
    follower_name = skf_chief.satellite_name  # kept for identifier naming pattern

    # --------- Baseline time grid & baseline state ---------
    t_skf = np.ravel(skf_chief.time) / 3600.0  # [hours]
    t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf_chief.pos)
    _,     skf_vel = data_processor.ensure_increasing(t_skf, skf_chief.vel)

    # --------- Assign Colors ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, time_hours, pos_3xn, vel_3xn, color)
    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = []
    datasets.append((base_label, t_skf, skf_pos, skf_vel, get_color(0)))

    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_obj_list = bsk_sim_data.sim_data
        if len(bsk_obj_list) <= chief_index:
            raise ValueError(f"Basilisk dataset for timestamp {ts} has no leader at index {chief_index}")

        bsk_chief = bsk_obj_list[chief_index]

        t = np.ravel(bsk_chief.time) / 3600.0  # [hours]
        t, bsk_pos = data_processor.ensure_increasing(t, bsk_chief.pos)
        _, bsk_vel = data_processor.ensure_increasing(t, bsk_chief.vel)

        datasets.append((legend_label, t, bsk_pos, bsk_vel, get_color(i)))

    # --------- Compute delta magnitudes vs baseline on baseline time grid ---------
    # Now:
    #   dpos_mag(t) = || r_sim(t) - r_base(t) ||
    #   dvel_mag(t) = || v_sim(t) - v_base(t) ||
    pos_diffs: dict[str, np.ndarray] = {}
    vel_diffs: dict[str, np.ndarray] = {}

    for label, t, pos_3xn, vel_3xn, _ in datasets:
        if label == base_label:
            pos_diffs[label] = np.zeros_like(t_skf)  # [m]
            vel_diffs[label] = np.zeros_like(t_skf)  # [m/s]
            continue

        # Interpolate sim onto baseline time grid (3 x n)
        if t.size != t_skf.size or not np.allclose(t, t_skf):
            pos_on_skf = data_processor.interp_3xn(t, pos_3xn, t_skf)
            vel_on_skf = data_processor.interp_3xn(t, vel_3xn, t_skf)
        else:
            pos_on_skf = pos_3xn
            vel_on_skf = vel_3xn

        dpos_vec = pos_on_skf - skf_pos
        dvel_vec = vel_on_skf - skf_vel

        pos_diffs[label] = np.linalg.norm(dpos_vec, axis=0) / 1000  # [km]
        vel_diffs[label] = np.linalg.norm(dvel_vec, axis=0)  # [m/s]

    # --------- Create figure with 2 stacked subplots (diffs only) ---------
    fig, (ax_posdiff, ax_veldiff) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    # Top: position delta magnitude
    for label, _, _, _, color in datasets:
        ax_posdiff.plot(t_skf, pos_diffs[label], label=label, color=color, linewidth=1.5)

    ax_posdiff.set_ylabel("ECI ||Δr|| (km)")
    ax_posdiff.set_title("Simulator Difference Position Delta Magnitude (||r - r_base||)")
    ax_posdiff.grid(True, alpha=0.3)

    # Bottom: velocity delta magnitude
    for label, _, _, _, color in datasets:
        ax_veldiff.plot(t_skf, vel_diffs[label], label=label, color=color, linewidth=1.5)

    ax_veldiff.set_xlabel("Time (hours)")
    ax_veldiff.set_ylabel("ECI ||Δv|| (m/s)")
    ax_veldiff.set_title("Simulator Difference Velocity Delta Magnitude (||v - v_base||)")
    ax_veldiff.grid(True, alpha=0.3)

    # ---- Single shared legend ----
    handles, labels = ax_posdiff.get_legend_handles_labels()
    ax_posdiff.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"ECI State Delta Magnitude Simulator Difference for the Leader Spacecraft\n"
        f"Leader: {chief_sat_name}"
    )

    plt_identifier = f"{main_plt_identifier}_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)


def plot_multi_sim_pos_sim_diff_mag(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    in ECI state *delta magnitude* differences for the leader spacecraft.

    Output: ONE figure with ONE plot:
      Position delta magnitude difference   ||r_sim - r_base||   [km]

    Notes:
        - Uses leader spacecraft at index 0.
        - Basilisk series are interpolated onto the baseline time grid.
        - Baseline is plotted as a horizontal zero line using base_label.
    """
    main_plt_identifier = "SimStateAbsMultiSimDiff"

    data_loader = DataLoader()
    data_processor = DataProcessor()

    # --------- Load baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_skf" #DFHSJKFHDSKHF
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No baseline datafiles found for timestamp '{skf_stamp}'")
    skf_sim_data = data_loader.load_sim_data_file(skf_files[0])

    # --------- Load Basilisk datasets ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []
    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")
        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Select leader spacecraft ---------
    chief_index = 0
    skf_obj_list = skf_sim_data.sim_data
    if len(skf_obj_list) <= chief_index:
        raise ValueError("Baseline dataset has no leader spacecraft at index 0.")

    skf_chief = skf_obj_list[chief_index]
    chief_sat_name = skf_chief.satellite_name
    follower_name = skf_chief.satellite_name

    # --------- Baseline time grid & state ---------
    t_skf = np.ravel(skf_chief.time) / 3600.0  # [hours]
    t_skf, skf_pos = data_processor.ensure_increasing(t_skf, skf_chief.pos)
    _, skf_vel = data_processor.ensure_increasing(t_skf, skf_chief.vel)

    # --------- Assign colors ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = []
    datasets.append((base_label, t_skf, skf_pos, skf_vel, get_color(0)))

    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        bsk_obj_list = bsk_sim_data.sim_data
        if len(bsk_obj_list) <= chief_index:
            raise ValueError(f"Basilisk dataset for timestamp {ts} has no leader at index {chief_index}")

        bsk_chief = bsk_obj_list[chief_index]
        t = np.ravel(bsk_chief.time) / 3600.0
        t, bsk_pos = data_processor.ensure_increasing(t, bsk_chief.pos)
        _, bsk_vel = data_processor.ensure_increasing(t, bsk_chief.vel)

        datasets.append((legend_label, t, bsk_pos, bsk_vel, get_color(i)))

    # --------- Compute position delta magnitudes ---------
    pos_diffs: dict[str, np.ndarray] = {}

    for label, t, pos_3xn, vel_3xn, _ in datasets:
        if label == base_label:
            pos_diffs[label] = np.zeros_like(t_skf)
            continue

        if t.size != t_skf.size or not np.allclose(t, t_skf):
            pos_on_skf = data_processor.interp_3xn(t, pos_3xn, t_skf)
        else:
            pos_on_skf = pos_3xn

        dpos_vec = pos_on_skf - skf_pos
        pos_diffs[label] = np.linalg.norm(dpos_vec, axis=0) / 1000  # [km]

    # --------- Create single-plot figure ---------
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(PLT_WIDTH, PLT_HEIGHT),
    )

    for label, _, _, _, color in datasets:
        ax.plot(t_skf, pos_diffs[label], label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("ECI ||Δr|| (km)")
    ax.set_title("Simulator Difference Position Delta Magnitude (||r - r_base||)")
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper right",
        frameon=True,
        ncol=1,
    )

    fig.suptitle(
        f"ECI State Delta Magnitude Simulator Difference for the Leader Spacecraft\n"
        f"Leader: {chief_sat_name}"
    )

    plt_identifier = f"{main_plt_identifier}_{follower_name}_vs_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)



def plot_groundtrack_multi_sim_comparison_start_stop(
    cfg: Config,
    skf_base_data_timestamp: str,
    bsk_data_entries: list[tuple[str, str]],
    base_label: str,
    start_plot_time_hours: float = 160.0,
    end_plot_time_hours=168.0,
    view_lon_min: float = 3.0,
    view_lon_max: float = 33.0,
    view_lat_min: float = 56.0,
    view_lat_max: float = 73.0,
) -> None:
    """
    Compare multiple Basilisk simulations against a single Skyfield baseline
    by plotting the leader spacecraft groundtrack (lat/lon) on a map.

    Structure matches plot_simulator_state_abs_multi_sim_diff:
      - Loads baseline Skyfield from timestamp
      - Loads N Basilisk datasets from timestamps + legend labels
      - Uses leader spacecraft at index 0
      - Applies the same time-window masking to every dataset
      - Uses one color per dataset across the whole figure
      - Single map figure (leader only)
      - Adds conditional_save_plot and conditional_show_plot

    Plot styling:
      - Skyfield baseline: solid line
      - Basilisk runs: dotted line (":")
      - Each dataset keeps a consistent color across the plot

    Map window defaults focus on Norway (+ nearby region).
    """
    main_plt_identifier = "GroundTrackMultiSim"

    if end_plot_time_hours is None:
        end_plot_time_hours = cfg.simulationDuration

    if end_plot_time_hours < start_plot_time_hours:
        raise ValueError(
            f"end_plot_time_hours ({end_plot_time_hours}) "
            f"must be >= start_plot_time_hours ({start_plot_time_hours})."
        )

    if view_lon_max <= view_lon_min or view_lat_max <= view_lat_min:
        raise ValueError(
            "Invalid map view window: require view_lon_max > view_lon_min and view_lat_max > view_lat_min."
        )

    data_loader = DataLoader()

    # Epoch for t = 0
    epoch, _, _ = get_simulation_time(cfg)

    # --------- Load Skyfield baseline ---------
    skf_stamp = f"{skf_base_data_timestamp}_skf" # REMEMBER TO SELECT THE REIGHT STRING ENDING !!!!
    skf_files = data_loader.get_datafiles_by_timestamp(skf_stamp)
    if len(skf_files) == 0:
        raise FileNotFoundError(f"No Skyfield datafiles found for timestamp '{skf_stamp}'")
    skf_sim_data = data_loader.load_sim_data_file(skf_files[0])

    # --------- Load Basilisk datasets (with legends) ---------
    bsk_sim_data_list: list[tuple[str, SimData, str]] = []  # (timestamp, simdata, legend)
    for (ts, legend_label) in bsk_data_entries:
        bsk_stamp = f"{ts}_bsk"
        bsk_files = data_loader.get_datafiles_by_timestamp(bsk_stamp)
        if len(bsk_files) == 0:
            raise FileNotFoundError(f"No Basilisk datafiles found for timestamp '{bsk_stamp}'")
        sim_data = data_loader.load_sim_data_file(bsk_files[0])
        bsk_sim_data_list.append((ts, sim_data, legend_label))

    # --------- Select leader spacecraft (index 0) ---------
    chief_index = 0
    if len(skf_sim_data.sim_data) <= chief_index:
        raise ValueError("Skyfield dataset has no leader spacecraft at index 0.")
    skf_chief = skf_sim_data.sim_data[chief_index]
    chief_sat_name = skf_chief.satellite_name

    # --------- Assign Colors (same approach as plot_alt_multi_sim_diff) ---------
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not default_colors:
        default_colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    def get_color(i: int) -> str:
        return default_colors[i % len(default_colors)]

    # dataset list: (label_for_plot, time_sec, x, y, z, color, linestyle)
    datasets: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str]] = []

    # Skyfield baseline (solid)
    datasets.append(
        (
            base_label,
            np.ravel(skf_chief.time),
            skf_chief.pos[0, :],
            skf_chief.pos[1, :],
            skf_chief.pos[2, :],
            get_color(0),
            "-",
        )
    )

    # Basilisk datasets (dotted)
    for i, (ts, bsk_sim_data, legend_label) in enumerate(bsk_sim_data_list, start=1):
        if len(bsk_sim_data.sim_data) <= chief_index:
            raise ValueError(f"Basilisk dataset for timestamp {ts} has no leader at index {chief_index}")
        bsk_chief = bsk_sim_data.sim_data[chief_index]
        datasets.append(
            (
                legend_label,
                np.ravel(bsk_chief.time),
                bsk_chief.pos[0, :],
                bsk_chief.pos[1, :],
                bsk_chief.pos[2, :],
                get_color(i),
                ":",  # dotted Basilisk
            )
        )

    # --------- Convert each dataset to lat/lon within requested time window ---------
    # Store in: [(label, lon_deg, lat_deg, color, linestyle)]
    tracks: list[tuple[str, np.ndarray, np.ndarray, str, str]] = []

    for label, t_sec, x, y, z, color, linestyle in datasets:
        t_hours = t_sec / 3600.0
        mask = (t_hours >= start_plot_time_hours) & (t_hours <= end_plot_time_hours)

        if not np.any(mask):
            continue

        t_sel = t_sec[mask]
        x_sel = x[mask]
        y_sel = y[mask]
        z_sel = z[mask]

        t_dt = [epoch + timedelta(seconds=float(t)) for t in t_sel]

        # ECI -> geodetic (lat [deg], lon [deg], alt [m])
        lat_deg, lon_deg, _ = pm.eci2geodetic(x_sel, y_sel, z_sel, t_dt)

        tracks.append((label, np.asarray(lon_deg), np.asarray(lat_deg), color, linestyle))

    if len(tracks) == 0:
        # Nothing to plot in the requested window
        return

    # --------- Create a single map figure (leader only) ---------
    fig = plt.figure(figsize=(PLT_WIDTH, PLT_HEIGHT))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent(
        [view_lon_min, view_lon_max, view_lat_min, view_lat_max],
        crs=ccrs.PlateCarree(),
    )

    ax.stock_img()
    ax.coastlines()
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.4)

    for label, lon_deg, lat_deg, color, linestyle in tracks:
        if lon_deg.size == 0:
            continue
        ax.plot(
            lon_deg,
            lat_deg,
            linewidth=1.5,
            linestyle=linestyle,
            color=color,
            label=label,
            transform=ccrs.Geodetic(),
        )

    ax.set_title(
        f"Groundtrack comparison (leader) — {chief_sat_name}\n"
        f"t ∈ [{start_plot_time_hours:.2f}, {end_plot_time_hours:.2f}] h"
    )
    ax.legend(loc="upper right")

    plt_identifier = f"{main_plt_identifier}_{chief_sat_name}"
    conditional_save_plot(cfg, fig, plt_identifier)
    conditional_show_plot(cfg)



########################################
# Plotting helper function definitions #
########################################

def conditional_save_plot(cfg: Config, fig: Figure, plt_identifier: str) -> None:
    """
    Save a matplotlib Figure to PLT_SAVE_FOLDER_PATH with a standardized filename iff cfg.save_plots == true.

    Filename: f"{data_timestamp}_{plt_identifier}.png"

    Args:
        fig: Matplotlib Figure object to save.
        data_timestamp: Timestamp string associated with the data (e.g. "20251106_003128").
        plt_identifier: Short identifier for the plot type/content (e.g. "pos_comp_sat1").
    """
    # Only save plots if cfg.save_plots == true
    if not cfg.save_plots:
        return
    
    # Get correct timestamp
    if not cfg.bypass_sim_to_plot:
        # plot results from this simulation 
        data_timestamp = cfg.timestamp_str  
    else:
        # plot results from a previous simulation
        data_timestamp = cfg.data_timestamp_to_plot

    # Ensure target directory exists
    PLT_SAVE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    filename = f"{data_timestamp}_{plt_identifier}.png"
    save_path = PLT_SAVE_FOLDER_PATH / filename

    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    logging.debug(f"Saved figure: {filename}")


def conditional_show_plot(cfg: Config) -> None:
    # Show plot only if cfg.show_plots == true
    if cfg.show_plots:
        plt.show()
    else:
        return
 

def quiet_plots() -> None:
    # Only show warnings and errors globally
    logging.basicConfig(level=logging.WARNING)

    # Matplotlib: silence backend + font-manager chatter
    mpl.set_loglevel("warning")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    # Pillow (PIL): silence PNG chunk debug like "STREAM b'IHDR'"
    #PngImagePlugin.debug = False
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)


def get_simulation_time(cfg: Config) -> tuple[datetime, int, int]:
    """
    Repurposed from 'SkyfieldSimulator.get_skf_simulation_time
    Parses and converts simulation time parameters from default config into skyfield-compatible types.

    RETURNS:
        startTime       (datetime) utc time
        duration        (float) seconds
        deltaT          (int) seconds
    """
    cfg_startTime = cfg.startTime
    cfg_duration = cfg.simulationDuration
    cfg_deltaT = cfg.s_set.deltaT

    # Parse simulation starttime and convert into a UTC datetime object
    try:
        startTime = datetime.strptime(cfg_startTime, "%d.%m.%Y %H:%M:%S").replace(tzinfo=timezone.utc)
    except:
        raise ValueError("Failed to convert config parameter 'startTime' to a datetime object.")
    
    # Convert cfg_duration float(hours) -> int(seconds)
    duration = int(3600 * cfg_duration) 
    if duration < (3600 * cfg_duration):
        raise ValueError("Type conversion float -> int for 'skf_duration' caused a reduction in its value!")

    # deltaT
    deltaT = int(cfg_deltaT)
    if deltaT < cfg_deltaT:
        raise ValueError("Type conversion float -> int for 'skf_deltaT' caused a reduction in its value!")
    
    return startTime, duration, deltaT