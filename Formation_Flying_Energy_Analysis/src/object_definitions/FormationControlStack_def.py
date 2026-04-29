from __future__ import annotations
from typing import TYPE_CHECKING

from Basilisk.architecture import messaging, sysModel

from object_definitions.Config_def import Config
from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle
if TYPE_CHECKING:
    # This is done to prevent the "low-level" environmental models being dependent 
    # on the "high-level" orchestrator
    from object_definitions.BasiliskSimulator_def import BasiliskSimulator 


class _FormationControlScheduler(sysModel.SysModel):
    def __init__(self, owner: "FormationControlStack"):
        super().__init__()
        self.owner = owner
        self.ModelTag = "FormationControlScheduler"

    def UpdateState(self, CurrentSimNanos: int) -> None:
        self.owner._update_state(CurrentSimNanos)

    def SelfInit(self):
        self.owner._self_init()

    def CrossInit(self):
        self.owner._cross_init()

    def Reset(self, CurrentSimNanos: int):
        self.owner._reset(CurrentSimNanos)



class FormationControlStack:
    def __init__(self,
                 sim: BasiliskSimulator,
                 cfg: Config,
                 scRuntimeBundles: list[SpacecraftRuntimeBundle]) -> None:
        
        self.scheduler = _FormationControlScheduler(self)
        self.sim = sim
        self.cfg = cfg
        self.scRuntimeBundles = scRuntimeBundles


        # Create Formaiton control task and add it to the formation control process
        self.formationControlTaskName = f"FormationControlTask"
        sim.formationControlProcess.addTask(sim.CreateNewTask(self.formationControlTaskName, sim.formCtrlRateNanos))

        
        # TODO: Add recorders owned by this class:

        # TODO: Message wiring


        # Add scheduler (TODO: and recorders) to task
        sim.AddModelToTask(self.formationControlTaskName, self.scheduler, 20)



    ##############################
    # SysModel Scheduler methods #
    ##############################

    def _modules(self):
        return []
    
    def _update_state(self, CurrentSimNanos: int) -> None:
        """
        Run all modules
        """
        pass


    def _self_init(self):
        for m in self._modules():
            if hasattr(m, "SelfInit"):
                m.SelfInit()


    def _cross_init(self):
        for m in self._modules():
            if hasattr(m, "CrossInit"):
                m.CrossInit()


    def _reset(self, CurrentSimNanos: int):
        for m in self._modules():
            if hasattr(m, "Reset"):
                m.Reset(CurrentSimNanos)


    #######################
    # Private GNC methods #
    #######################
