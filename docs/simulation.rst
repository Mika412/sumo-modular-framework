Simulation
===========


To create a custom simulation, these files are required to be in the root location of the simulation:

* **osm.net.xml:** Defines the properties of the map. 
* **generated/routes.xml:** Specifies the vehicle routes
* **generated/flows.xml:** Specifies the flows for each routes
* **polys.add.xml:**  Defines the cell shapes
* **districts.taz.xml:** Specifies which edges belong to which cell
* **config.ini:** Defines the configuration of the simulation
* **osm.poly.add.xml (optional):** Contains the background information
* **data/induction_loops.add.xml (optional):** Simulation induction loop output


Map
^^^

| The map specifies all the properties of the simulated map environment. It defines the roads, crossroads, vehicle permissions, etc.
| This map can be created by different means. One is by using the *netedit* tool, which is a GUI program that allows you to manually create and edit the map. Another one is to generate from OpenStreetsMap. The easiest solution is to use the `osmWebWizard.py script <https://sumo.dlr.de/docs/Tools/Import/OSM.html>`_.

Routes and Flows
^^^^^^^^^^^^^^^^

Emissions
^^^^^^^^^

Configuration
^^^^^^^^^^^^^

