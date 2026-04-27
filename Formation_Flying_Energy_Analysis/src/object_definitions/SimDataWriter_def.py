from dataclasses import dataclass
from object_definitions.Config_def import Config
from object_definitions.SpacecraftRuntimeBundle_def import SpacecraftRuntimeBundle
from object_definitions.SimData_def import (SpacecraftSimData, MissionSimData)

class SimDataWriter:
    def __init__(self, cfg: Config, scSimDataList: list[SpacecraftSimData], missionSimData = None) -> None:
        
        self.output_data_save_dir = cfg.output_data_save_dir # All output data will be stored in this folder
        self.mc_enabled = cfg.mc_enabled
        self.run_idx = cfg.run_idx # Needed for naming data file
        self.scSimDataList = scSimDataList
        self.missionSimData = missionSimData
        self.numSc = len(scSimDataList)

    
    def write_data_to_files(self) -> None:
        if self.mc_enabled:
            self._write_reduced_data_to_files()
        else:
            self._write_full_data_to_files()

    
    def _write_reduced_data_to_files(self) -> None:
        """
        Write a subset of the data 'self.simData' to file 
        """
        pass

    
    def _write_full_data_to_files(self) -> None:
        pass