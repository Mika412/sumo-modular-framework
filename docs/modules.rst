Modules
=======

The project work works in a modular manner, where each module implements a desired functionality and should be only used
when needed. During the simulation environment creating, a list of modules are passed as one of the parameters. These
modules must implement a specific structure defined by BaseModule class. The
environment takes care of calling the module base functions, but it is the modules responsability of have the desired
functionality implemented.
Currently the Framework comes with following modules:

* **CellsModule:** Creates a cell matrix, and what edges a cell contains. Cells/edge relation is created based on
  *.taz.xml* file.
* **EmissionsModule:** Tracks all the cell emissions values. Simulates the cell propagation and dissipation over time.
* **EmissionsRendererModule:** Renders the emission matrix.
* **InductionLoopsModule:** Tracks all the vehicles that passed though the Induction Loops
* **TrackingModule:** Tracks statistical information of the simulation.


Creating a module
^^^^^^^^^^^^^^^^^

A custom module needs to implement the following class::

    class BaseModule:
    
        @property
        @abstractmethod
        def variable_name(self):
            """A simple name to identify the module"""
            pass

        @abstractmethod
        def step(self, timestep):
            """Called at every step of the simulation"""
            pass
    
        @abstractmethod
        def reset(self):
            """Called everytime the environment is reset"""
            pass

An example of a module that tracks the number of the vehicles per timestep would be::
    
    import traci
    from BaseModule import BaseModule

    class VehicleTracker(BaseModule):
        __init__(self):
            # Recommended that each module tracks the traci instance. 
            # Needed for running multiple simulations simultaneously
            self._traci = traci 
            self.vehicles = {}

        @property
        def variable_name(self):
            return "vehicle_tracker"

        def step(self, timestep):
            self.vehicles[timestep] = self._traci._vehicle.getIDCount()

        def reset(self):
            self.vehicles = {}
