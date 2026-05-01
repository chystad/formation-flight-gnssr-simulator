import h5py
import logging
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from numpy.typing import NDArray
from dataclasses import dataclass

from Basilisk.utilities import macros

from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle

# Global definition of data save folder path
OUTPUT_DATA_SAVE_DIR = Path('Formation_Flying_Energy_Analysis/output_data/sim_data')


@dataclass
class SampledData:
    data: NDArray[Any]
    dt: float
    n_samples: int

# TODO
@dataclass 
class MissionSimData:
    TODO: bool

# TODO
@dataclass 
class SpacecraftSimData:
    # FSW-owned data
    r_BN_N: SampledData # (n, 3) [m] B position relative to N, exporessed in N frame
    v_BN_N: SampledData # (n, 3) [m/s] B velocity relative to N, expressed in N frame

    sigma_BN: SampledData   # (n, 3) [MRP] Attitude of B relative to N
    omega_BN_B: SampledData # (n, 3) [rad/s] Angular rate of B relative to N, expressed in B frame

    sigma_RN: SampledData   # (n, 3) [MRP] Desired attitude, R relative to N
    omega_RN_N: SampledData # (n, 3) [rad/s] Desired angular rate, R relative to N, expressed in N frame

    sigma_BR: SampledData   # (n, 3) [MRP] Attitude tracking error of B relative to R
    omega_BR_B: SampledData # (n, 3) [rad/s] Angular rate tracking error of B relative to R, expressed in B frame

    cmdTorqueBody: SampledData  # (n,      3) [Nm] Torque command from FSW, expressed in Body
    cmdMotorTorque: SampledData # (n, numRWs) [Nm] Torque command from FSW, expressed in RW frame

    # Dynamics-owned data
    thrustForce_B: SampledData        # (n, 3) [N] Thrust force vector, expressed in B frame
    thrustTorquePntB_B: SampledData   # (n, 3) [Nm] Thrust torque about point B, expressed in B frame
    thrustBlowDownFactor: SampledData # (n,) [frac/%???] Current thrust percentage due to tank blow down
    ispBlowDownFactor: SampledData    # (n,) [frac/%???] Current Isp percentage due to tank blow down

    fuelMass: SampledData # (n,) [kg] Fuel mass

    rwOmega: SampledData    # (n, numRWs) [rad/s] Wheel speed
    rwUCurrent: SampledData # (n, numRWs) [Nm] Motor torque
    rwNetPower: SampledData # (n, numRWs) [W] RW power used/generated ( < 0 => Consume power)

    storageLevel: SampledData    # (n,) [Ws] Battery stored charge
    currentNetPower: SampledData # (n,) [W] Net power received/drained from the battery

    obcNetPower: SampledData        # (n,) [W] Net power used/generated
    solarPanelNetPower: SampledData # (n, numSPs) [W] Net power used/generated

@dataclass
class FormationFollowerStatus:
    sat_idx: int
    desired_along_track_m: float
    radial_error_m: float
    along_track_error_m: float
    cross_track_error_m: float
    pos_error_norm_m: float
    vel_error_norm_mps: float
    inside_tolerance: bool
    dwell_time_s: float
    achieved: bool
    burn_required: bool
    burn_direction_rtn: NDArray[np.float64]
    burn_attitude_mrp: NDArray[np.float64]
    thrust_on_time_s: float



class SimData:
    def __init__(self) -> None:
        pass


    #########################
    # Public Helper methods #
    #########################
    
    def pull_every_spacecraft_data(self, 
                                   scRuntimeBundles: list[SpacecraftRuntimeBundle | None]
                                   ) -> list[SpacecraftSimData]:
        """
        Pull data from each spacecraft

        Returns:
            list[SpacecraftSimData]: data from list index #i corresponds to spacecraft #sat_idx data
        """

        # Raise error if scRuntimeBundles isn't initialized
        assert scRuntimeBundles is not None
        
        # Local data containers for spacecraft and mission data
        scSimDataList: list[SpacecraftSimData] = []
        
        # Extract per-spacecraft data from recorders
        for i in range(len(scRuntimeBundles)):
            scRuntimeBundle = scRuntimeBundles[i]

            # Validate satellite order
            assert scRuntimeBundle is not None
            if i != scRuntimeBundle.sat_idx:
                raise ValueError(f"Index mismatch between element #{i} in scRuntimeBundles and it satellite index #{scRuntimeBundle.sat_idx}")

            scSimData = self._pull_single_spacecraft_data(scRuntimeBundle)
            scSimDataList.append(scSimData)

            # self._DEBUG_print_spacecraft_sim_data_field_sizes(scRuntimeBundle.sat_idx, scSimData)

        return scSimDataList
    


    ##########################
    # Private Helper methods #
    ##########################

    def _pull_single_spacecraft_data(self, scRuntimeBundle: SpacecraftRuntimeBundle) -> SpacecraftSimData:
        """
        Pull all relevant data fields from the recorders in SpacecraftRuntimeBundle.

        Returns:
            SpacecraftSimData: Data container for one spacecraft
        """
        
        dynModel = scRuntimeBundle.dynModel
        fsw = scRuntimeBundle.fsw

        # ---------------------------------------------------------
        # FSW-owned data extraction as tuple (data, dt, n)
        # ---------------------------------------------------------

        # Ensure that all FSW recorders described in FswStack._setup_fsw_recorders()
        # are available before extracting data.
        assert fsw.navTransRecorder is not None
        assert fsw.navAttRecorder is not None
        assert fsw.attRefRecorder is not None
        assert fsw.attErrRecorder is not None
        assert fsw.cmdTorqueRecorder is not None
        assert fsw.rwMotorTorqueRecorder is not None
        
        # Translational states
        r_BN_N_data = SampledData(
            np.asarray(fsw.navTransRecorder.r_BN_N),
            fsw.navTransRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.navTransRecorder.r_BN_N))
        v_BN_N_data = SampledData(
            np.asarray(fsw.navTransRecorder.v_BN_N),
            fsw.navTransRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.navTransRecorder.v_BN_N))

        # Spacecraft attitude states
        sigma_BN_data = SampledData(
            np.asarray(fsw.navAttRecorder.sigma_BN),
            fsw.navAttRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.navAttRecorder.sigma_BN))
        omega_BN_B_data = SampledData(
            np.asarray(fsw.navAttRecorder.omega_BN_B),
            fsw.navAttRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.navAttRecorder.omega_BN_B))

        # Desired attitude states
        sigma_RN_data = SampledData(
            np.asarray(fsw.attRefRecorder.sigma_RN),
            fsw.attRefRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.attRefRecorder.sigma_RN))
        omega_RN_N_data = SampledData(
            np.asarray(fsw.attRefRecorder.omega_RN_N),
            fsw.attRefRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.attRefRecorder.omega_RN_N))

        # Attitude tracking errors
        sigma_BR_data = SampledData(
            np.asarray(fsw.attErrRecorder.sigma_BR),
            fsw.attErrRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.attErrRecorder.sigma_BR))
        omega_BR_B_data = SampledData(
            np.asarray(fsw.attErrRecorder.omega_BR_B),
            fsw.attErrRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.attErrRecorder.omega_BR_B))

        # Control outputs
        cmdTorqueBody_data = SampledData(
            np.asarray(fsw.cmdTorqueRecorder.torqueRequestBody),
            fsw.cmdTorqueRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.cmdTorqueRecorder.torqueRequestBody))
        cmdMotorTorque_data = SampledData(
            np.asarray(fsw.rwMotorTorqueRecorder.motorTorque)[:, :dynModel.numRWs], # default size: (n, 36), reduce to (n, numRWs)
            fsw.rwMotorTorqueRecorder_RateNanos * macros.NANO2SEC,
            len(fsw.rwMotorTorqueRecorder.motorTorque))



        # ---------------------------------------------------------
        # Dynamics-owned data extraction as tuple (data, dt, n)
        # ---------------------------------------------------------

        assert dynModel.thrusterStateRecorder is not None
        assert dynModel.fuelTankStateRecorder is not None
        assert dynModel.batteryStateRecorder is not None
        assert dynModel.obcPowerSinkRecorder is not None
        assert len(dynModel.rwStateRecorders) == dynModel.numRWs
        assert len(dynModel.rwPowerRecorders) == dynModel.numRWs
        assert len(dynModel.solarPanelPowerRecorders) == dynModel.numSPs

        # Thruster state
        thrustForce_B_data = SampledData(
            np.asarray(dynModel.thrusterStateRecorder.thrustForce_B),
            dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.thrusterStateRecorder.thrustForce_B))
        thrustTorquePntB_B_data = SampledData(
            np.asarray(dynModel.thrusterStateRecorder.thrustTorquePntB_B),
            dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.thrusterStateRecorder.thrustTorquePntB_B))
        thrustBlowDownFactor_data = SampledData(
            np.asarray(dynModel.thrusterStateRecorder.thrustBlowDownFactor),
            dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.thrusterStateRecorder.thrustBlowDownFactor))
        ispBlowDownFactor_data = SampledData(
            np.asarray(dynModel.thrusterStateRecorder.ispBlowDownFactor),
            dynModel.thrusterStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.thrusterStateRecorder.ispBlowDownFactor))
        

        # Fuel tank state
        fuelMass_data = SampledData(
            np.asarray(dynModel.fuelTankStateRecorder.fuelMass),
            dynModel.fuelTankStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.fuelTankStateRecorder.fuelMass))

        # Reaction wheel states, one array per RW
        rwOmega_data = SampledData(
            np.asarray([rec.Omega for rec in dynModel.rwStateRecorders]).T,
            dynModel.rwStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.rwStateRecorders[0].Omega) if dynModel.numRWs > 0 else 0)
        rwUCurrent_data = SampledData(
            np.asarray([rec.u_current for rec in dynModel.rwStateRecorders]).T,
            dynModel.rwStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.rwStateRecorders[0].u_current) if dynModel.numRWs > 0 else 0)

        # Reaction wheel power consumption, one array per RW
        rwNetPower_data = SampledData(
            np.asarray([rec.netPower for rec in dynModel.rwPowerRecorders]).T,
            dynModel.rwPowerRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.rwPowerRecorders[0].netPower) if dynModel.numRWs > 0 else 0)

        # Battery state
        storageLevel_data = SampledData(
            np.asarray(dynModel.batteryStateRecorder.storageLevel),
            dynModel.batteryStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.batteryStateRecorder.storageLevel))
        currentNetPower_data = SampledData(
            np.asarray(dynModel.batteryStateRecorder.currentNetPower),
            dynModel.batteryStateRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.batteryStateRecorder.currentNetPower))

        # OBC power sink
        obcNetPower_data = SampledData(
            np.asarray(dynModel.obcPowerSinkRecorder.netPower),
            dynModel.obcPowerSinkRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.obcPowerSinkRecorder.netPower))

        # Solar panel power generation, one array per solar panel
        solarPanelNetPower_data = SampledData(
            np.asarray([rec.netPower for rec in dynModel.solarPanelPowerRecorders]).T,
            dynModel.solarPanelPowerRecorder_RateNanos * macros.NANO2SEC,
            len(dynModel.solarPanelPowerRecorders[0].netPower) if dynModel.numSPs > 0 else 0)
        

        scSimData = SpacecraftSimData(
            r_BN_N=r_BN_N_data,
            v_BN_N=v_BN_N_data,
            sigma_BN=sigma_BN_data,
            omega_BN_B=omega_BN_B_data,
            sigma_RN=sigma_RN_data,
            omega_RN_N=omega_RN_N_data,
            sigma_BR=sigma_BR_data,
            omega_BR_B=omega_BR_B_data,
            cmdTorqueBody=cmdTorqueBody_data,
            cmdMotorTorque=cmdMotorTorque_data,
            thrustForce_B=thrustForce_B_data,
            thrustTorquePntB_B=thrustTorquePntB_B_data,
            thrustBlowDownFactor=thrustBlowDownFactor_data,
            ispBlowDownFactor=ispBlowDownFactor_data,
            fuelMass=fuelMass_data,
            rwOmega=rwOmega_data,
            rwUCurrent=rwUCurrent_data,
            rwNetPower=rwNetPower_data,
            storageLevel=storageLevel_data,
            currentNetPower=currentNetPower_data,
            obcNetPower=obcNetPower_data,
            solarPanelNetPower=solarPanelNetPower_data,
        )
        logging.debug(f"[DATA] Extracted data from spacecraft #{scRuntimeBundle.sat_idx}")
        return scSimData




    ##########################
    # Private DEEBUG methods #
    ##########################

    @staticmethod
    def _DEBUG_print_spacecraft_sim_data_field_sizes(sat_idx: int, 
                                                     scSimData: SpacecraftSimData
                                                     ) -> None:
        logging.debug(f"""
[DATA] Spacecraft #{sat_idx} data shapes

FSW-owned data:
  r_BN_N:              {np.shape(scSimData.r_BN_N.data)}
  v_BN_N:              {np.shape(scSimData.v_BN_N.data)}
  sigma_BN:            {np.shape(scSimData.sigma_BN.data)}
  omega_BN_B:          {np.shape(scSimData.omega_BN_B.data)}
  sigma_RN:            {np.shape(scSimData.sigma_RN.data)}
  omega_RN_N:          {np.shape(scSimData.omega_RN_N.data)}
  sigma_BR:            {np.shape(scSimData.sigma_BR.data)}
  omega_BR_B:          {np.shape(scSimData.omega_BR_B.data)}
  cmdTorqueBody:       {np.shape(scSimData.cmdTorqueBody.data)}
  cmdMotorTorque:      {np.shape(scSimData.cmdMotorTorque.data)}

Dynamics-owned data:
  thrustForce_B:       {np.shape(scSimData.thrustForce_B.data)}
  thrustBlowDownFactor:{np.shape(scSimData.thrustBlowDownFactor.data)}
  ispBlowDownFactor:   {np.shape(scSimData.ispBlowDownFactor.data)}
  thrustTorquePntB_B:  {np.shape(scSimData.thrustTorquePntB_B.data)}
  fuelMass:            {np.shape(scSimData.fuelMass.data)}
  rwOmega:             {np.shape(scSimData.rwOmega.data)}
  rwUCurrent:          {np.shape(scSimData.rwUCurrent.data)}
  rwNetPower:          {np.shape(scSimData.rwNetPower.data)}
  storageLevel:        {np.shape(scSimData.storageLevel.data)}
  currentNetPower:     {np.shape(scSimData.currentNetPower.data)}
  obcNetPower:         {np.shape(scSimData.obcNetPower.data)}
  solarPanelNetPower:  {np.shape(scSimData.solarPanelNetPower.data)}
""")