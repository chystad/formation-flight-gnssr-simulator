#
#  ISC License
#
#  Copyright (c) 2021, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, TypeAlias

import logging
import itertools
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import macros, fswSetupThrusters, RigidBodyKinematics as rbk
from Basilisk.fswAlgorithms import spacecraftReconfig, formationBarycenter

from object_definitions.Config_def import Config
from object_definitions.Satellite_def import Satellite
from object_definitions.BasiliskDynamicsModel_def import BasiliskDynamicsModel
if TYPE_CHECKING:
    # This is done to prevent the "low-level" environmental models being dependent 
    # on the "high-level" orchestrator
    from object_definitions.FswStack_def import FswStack
    from object_definitions.BasiliskSimulator_def import BasiliskSimulator 


class _ThrCmdSafeWriter(sysModel.SysModel):
    def __init__(self, owner: "FormationControlStack"):
        super().__init__()
        self.owner = owner
        self.ModelTag = f"ThrCmdSafeWriter_{owner.sat_idx}"

    def UpdateState(self, CurrentSimNanos: int) -> None:
        self.owner._copy_or_zero_thr_cmd(CurrentSimNanos)



class FormationControlStack:
    def __init__(self,
                 sim: BasiliskSimulator,
                 sat: Satellite,
                 sat_idx: int,
                 dynModel: BasiliskDynamicsModel,
                 fsw: FswStack,
                 ) -> None:
        
        self.sim = sim
        self.satellite = sat
        self.sat_idx = sat_idx
        self.dynModel = dynModel
        self.fsw = fsw
        self.form_type = sim.cfg.form_type
        self.relNavEnabled = False
        self.thrCmdSafeWriter = _ThrCmdSafeWriter(self)
        self.logTag = f"FC{sat_idx}"

        # Exposed outputs
        self.form_att_ref_out_msg = None
        self.form_thr_cmd_out_msg = None

        # Internal messages
        self.fsw_thruster_config_msg = None

        # Create Formation control task and add it to the formation control process
        assert sim.formCtrlProcesses[sat_idx] is not None
        self.formationControlTaskName = f"FormationControlTask_{sat_idx}"
        sim.formCtrlProcesses[sat_idx].addTask(sim.CreateNewTask(self.formationControlTaskName, sim.formCtrlRateNanos)) # type: ignore

        # Initialize formation control modules
        self.formCtrl = spacecraftReconfig.spacecraftReconfig()
        self.relNav: Optional[formationBarycenter.FormationBarycenter] = None

        # Check if the selected formation type requires the relative navigation module
        if (self.form_type == "cpo") or (self.form_type == "cc"):
            self.relNavEnabled = True

        # Initialize all models, recorders and messages
        self._setup_thruster_cmd_msg()
        self._setup_thruster_config_msg()
        self._setup_formation_control()
        self._setup_desired_OE_difference()

        # Schedule all models
        sim.AddModelToTask(self.formationControlTaskName, self.formCtrl, 20)
        sim.AddModelToTask(self.formationControlTaskName, self.thrCmdSafeWriter, 10) # Must run later than formCtrl



    #########################
    # Public helper methods #
    ######################### 

    def connect_chief_trans_to_form_ctrl(self, fswChief: FswStack) -> None:
            """
            Connect the chief translational states to the spacecraftReconfig model

            Args:
                fswChief (FswStack): The chief's FSW stack
            """
            self.formCtrl.chiefTransInMsg.subscribeTo(fswChief.nav.transOutMsg)



    def _setup_formation_control(self) -> None:
        
        # setup formation control module
        self.formCtrl.ModelTag = f"formationControl_{self.sat_idx}"

        assert self.sim.envModel.gravFactory is not None
        assert self.fsw_thruster_config_msg is not None
        self.formCtrl.deputyTransInMsg.subscribeTo(self.fsw.nav.transOutMsg)
        self.formCtrl.attRefInMsg.subscribeTo(self.fsw.guid.attRefOutMsg)
        self.formCtrl.thrustConfigInMsg.subscribeTo(self.fsw_thruster_config_msg)
        self.formCtrl.vehicleConfigInMsg.subscribeTo(self.dynModel.vehicle_config_out_msg)
        self.formCtrl.mu = self.sim.envModel.gravFactory.gravBodies["earth"].mu  # [m^3/s^2]
        self.formCtrl.attControlTime = 400  # [s]
        
        # Expose spacecraftReconfig outputs directly to FswStack
        self.form_att_ref_out_msg = self.formCtrl.attRefOutMsg
        # self.form_thr_cmd_out_msg = self.formCtrl.onTimeOutMsg

        # connect a blank chief message (temporary until connection to real chief)
        chiefData = messaging.NavTransMsgPayload()
        chiefMsg = messaging.NavTransMsg().write(chiefData)
        self.formCtrl.chiefTransInMsg.subscribeTo(chiefMsg)


    def _setup_thruster_config_msg(self) -> None:
        """
        Imports the thrusters configuration information.
        NOTE: Must be run before '_setup_formation_control'
        """
        assert self.dynModel.thrusterFactory is not None

        fswSetupThrusters.clearSetup()
        for key, th in self.dynModel.thrusterFactory.thrusterList.items():
            loc_B_tmp = list(itertools.chain.from_iterable(th.thrLoc_B))
            dir_B_tmp = list(itertools.chain.from_iterable(th.thrDir_B))
            fswSetupThrusters.create(loc_B_tmp, dir_B_tmp, th.MaxThrust)
        self.fsw_thruster_config_msg = fswSetupThrusters.writeConfigMessage()


    def _setup_thruster_cmd_msg(self) -> None:
        """
        Writes a safe thruster command message corresponding to 0 thrust
        """
        self.form_thr_cmd_out_msg = messaging.THRArrayOnTimeCmdMsg()
        payload = messaging.THRArrayOnTimeCmdMsgPayload()
        payload.OnTimeRequest = [0.0]
        self.form_thr_cmd_out_msg.write(payload)


    def _copy_or_zero_thr_cmd(self, CurrentSimNanos: int) -> None:
        payload = messaging.THRArrayOnTimeCmdMsgPayload()
        
        try:
            # logging.debug(f"[{self.logTag}] Trying to read thrust command message (isWritten: {self.formCtrl.onTimeOutMsg.isWritten()})...")
            raw = self.formCtrl.onTimeOutMsg.read()
            payload.OnTimeRequest = [float(x) for x in raw.OnTimeRequest]
        except Exception:
            logging.debug(f"[{self.logTag}] Failed to read thrust command message")
            payload.OnTimeRequest = [0.0]

        assert self.form_thr_cmd_out_msg is not None
        self.form_thr_cmd_out_msg.write(payload, CurrentSimNanos)


    def _setup_desired_OE_difference(self) -> None:
        """
        Calculates and sets the desired classic orbital element difference dependinng on the selected formation type

        The spacecraftReconfig module expects the desired classic orbital element difference to be on the following format:
            [da, de, di, dOmega, domega, dM], 
        Where 'da' is normalized to become  dimentionless 
        """
        
        if self.sim.cfg.form_type == "cat":
            
            # Don't assign  desired OED for chief spacecraft
            if self.sat_idx == 0:
                return

            desiredSeparation = self.sim.cfg.cat_const_separation

            # TODO: Calculate the desired OED to get the desired separation given circular cheif orbit

            self.formCtrl.targetClassicOED = [
                0.0, # da/a
                0.0, # de
                0.0, # di
                0.0, # dOmega
                0.0, # domega
                -0.01*self.sat_idx] # dM

        else: 
            raise ValueError(f"Formation types other than 'constant along-track separation has not yet been implemented")
        