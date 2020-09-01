import csv
from enum import Enum
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from .BaseModule import BaseModule
from environment.modules.LimitedEmissionsModule import EmissionType
import scipy.ndimage.filters as filters
import matplotlib

matplotlib.use("Agg")

# if 'SUMO_HOME' in os.environ:
sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import traci


class TrackingModule(BaseModule):
    def __init__(
        self,
        cell_module,
        emissions_module,
        output_dir,
        sample_every,
        plot_title,
        tracking_cells=[],
    ):
        super().__init__()

        self._traci = traci
        self._cells = cell_module
        self._emissions = emissions_module

        self.output_dir = output_dir

        self.sample_every_steps = sample_every
        self.plot_title = plot_title
        self.tracking_cells = tracking_cells

        self.total_emissions = 0
        self.closed_cells = 0
        self.travel_time = 0
        self.waiting_time = 0
        self.current_episode = -1

        self.sampled_timestep = []
        self.total_emissions_history = []
        self.closed_cells_history = []
        self.travel_time_history = []
        self.waiting_time_history = []
        self.highest_emission_cell_value_history = []

        self.loaded_vehicles_history = []
        self.departed_vehicles_history = []
        self.arrived_vehicles_history = []
        self.departed_vehicles = 0
        self.loaded_vehicles = 0
        self.arrived_vehicles = 0

        self.cell_emissions_history = [[], []]

        self.number_of_file = 1
        with open(self.output_dir + "simulated_results.csv", "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            row = [
                "timestep",
                "Aggregated Pollution",
                "Max Cell Value",
                "Travel Time",
                "Waiting time",
                "Closed Cells",
                "Loaded Vehicles",
                "Departed Vehicles",
                "Arrived Vehicles",
            ]
            writer.writerow(row)

    @property
    def variable_name(self):
        return "tracking"

    def step(self, timestep):
        is_sample_step = timestep % self.sample_every_steps == 0
        self.updateEmissions(is_sample_step)
        self.updateClosedCells(is_sample_step)
        self.updateTravelTime(is_sample_step)
        self.updateVehicles(is_sample_step)
        if is_sample_step:
            self.sampled_timestep.append(timestep / 3600)
            self.closed_cells_history.append(len(self._cells.closed_cells))
            self.plot_data()

    def updateEmissions(self, sample_step):
        if sample_step:
            emissions_matrix = self._emissions.get_emissions_type_matrix(
                EmissionType.NOx)
            # Update emissions
            self.total_emissions_history.append(np.sum(emissions_matrix))

            self.highest_emission_cell_value_history.append(
                np.amax(emissions_matrix))

    def updateClosedCells(self, sample_step):
        # Update closed cells
        self.closed_cells += len(self._cells.closed_cells)

    def updateTravelTime(self, sample_step):
        # Update closed cells
        for edge in self._cells.edge_to_cells.keys():
            self.travel_time += self._traci.edge.getTraveltime(edge)
            self.waiting_time += self._traci.edge.getWaitingTime(edge)
        if sample_step:
            self.travel_time_history.append(self.travel_time)
            self.waiting_time_history.append(self.waiting_time)
            self.travel_time = 0
            self.waiting_time = 0

    def updateVehicles(self, sample_step):
        # Update closed cells
        self.loaded_vehicles += self._traci.simulation.getLoadedNumber()
        self.departed_vehicles += self._traci.simulation.getDepartedNumber()
        self.arrived_vehicles += self._traci.simulation.getArrivedNumber()
        if sample_step:
            self.loaded_vehicles_history.append(self.loaded_vehicles)
            self.departed_vehicles_history.append(self.departed_vehicles)
            self.arrived_vehicles_history.append(self.arrived_vehicles)
            self.departed_vehicles = 0
            self.loaded_vehicles = 0
            self.arrived_vehicles = 0

    def plot_data(self):
        if len(self.sampled_timestep) == 0:
            return
        # fig = plt.figure(constrained_layout=True)
        fig = plt.figure()
        fig.set_figheight(30)
        fig.set_figwidth(15)
        fig.suptitle(self.plot_title)
        gs = fig.add_gridspec(6, 2)

        multiply_by = 1
        # TOTAL EMISSIONS_REWARD

        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(
            self.sampled_timestep,
            [i * multiply_by for i in self.total_emissions_history],
            color="b",
        )
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("NOx Aggregated Pollution")
        ax1.set_ylim(bottom=0, top=10000)

        # HIGHEST CELL VALUE

        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(
            self.sampled_timestep,
            [
                i * multiply_by
                for i in self.highest_emission_cell_value_history
            ],
            color="b",
        )
        ax2.set_xlabel("Hour")
        ax2.set_ylabel("Max Cell Value")
        ax2.set_ylim(bottom=0, top=300)

        # TOTAL HALTED REWARD
        ax3 = fig.add_subplot(gs[2, :])
        ax3.plot(
            self.sampled_timestep,
            [i * multiply_by for i in self.travel_time_history],
            color="b",
        )
        ax3.set_xlabel("Hour")
        ax3.set_ylabel("Travel time")
        ax3.set_ylim(bottom=0, top=15e8)

        # TOTAL HALTED REWARD
        ax4 = fig.add_subplot(gs[3, :])
        ax4.plot(
            self.sampled_timestep,
            [i * multiply_by for i in self.waiting_time_history],
            color="b",
        )
        ax4.set_xlabel("Hour")
        ax4.set_ylabel("Waiting time")
        ax4.set_ylim(bottom=0, top=7e7)

        # CLOSED CELL HISTORY
        ax4 = fig.add_subplot(gs[4, :])
        ax4.plot(self.sampled_timestep, self.closed_cells_history, color="b")
        ax4.set_xlabel("Hour")
        ax4.set_ylabel("Closed cells")
        ax4.set_ylim(bottom=0)

        ax5 = fig.add_subplot(gs[5, :])
        ax5.plot(self.sampled_timestep,
                 self.loaded_vehicles_history,
                 color="b",
                 label="loaded")
        ax5.plot(self.sampled_timestep,
                 self.departed_vehicles_history,
                 color="r",
                 label="departed")
        ax5.plot(self.sampled_timestep,
                 self.arrived_vehicles_history,
                 color="g",
                 label="arrived")
        ax5.legend(loc='best')
        ax5.set_xlabel("Hour")
        ax5.set_ylabel("Vehicles")
        ax5.set_ylim(bottom=0)

        fig.savefig(self.output_dir + "tracking_scores" +
                    str(self.number_of_file) + ".png")
        plt.close(fig)

        self.write_data()

    def write_data(self):
        with open(self.output_dir + "simulated_results.csv", "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")

            writer.writerow([
                self.sampled_timestep[-1], self.total_emissions_history[-1],
                self.highest_emission_cell_value_history[-1],
                self.travel_time_history[-1], self.waiting_time_history[-1],
                self.closed_cells_history[-1],
                self.loaded_vehicles_history[-1],
                self.departed_vehicles_history[-1],
                self.arrived_vehicles_history[-1]
            ])

    def reset(self):
        # self.plot_data()
        self.sampled_timestep = []
        self.total_emissions_history = []
        self.highest_emission_cell_value_history = []
        self.closed_cells_history = []
        self.loaded_vehicles_history = []
        self.departed_vehicles_history = []
        self.arrived_vehicles_history = []
        self.travel_time = 0
        self.waiting_time = 0
        self.travel_time_history = []
        self.waiting_time_history = []
        self.current_episode += 1
        self.total_emissions = 0
        self.closed_cells = 0
        self.travel_time = 0
        self.departed_vehicles = 0
        self.loaded_vehicles = 0

        self.number_of_file += 1
