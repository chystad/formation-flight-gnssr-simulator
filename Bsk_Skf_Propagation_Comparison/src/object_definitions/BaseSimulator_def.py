from abc import ABC, abstractmethod
from object_definitions.Config_def import Config

"""
=========================================================================================================
This modules defines the BaseSimulator abstract class.
Its purpose is to provide a common interface for all simulator implementations by defining a set of predefined methods.
All simulators accross every simulation framework must inherit this class and implement its methods.
=========================================================================================================
"""

class BaseSimulator(ABC):
    """Abstract base class for all simulator implementations"""

    def __init__(self, cfg: Config):
        """Inherit the simulation configuration"""
        self.cfg = cfg
    
    @abstractmethod
    def setup(self):
        """Initialize the simulator given config (create simulation objects, set initial conditions, attach dynamics, etc.)"""
        pass

    @abstractmethod
    def run(self):
        """Run the simulation"""
        pass