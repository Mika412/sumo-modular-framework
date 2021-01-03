import math
import sys

import numpy as np
from gym import spaces

sys.path.append(".")

import configparser

from environment.core.BaseRLEnv import SumoRLBaseEnvironment
from environment.modules.CellsModule import CellsModule
from environment.modules.EmissionsModule import EmissionsModule, EmissionType

class CustomEnv(SumoRLBaseEnvironment):
    def __init__(
        self,
        env_dir,
        use_gui=False,
        num_seconds=20000,
        start_at=0,
        action_every_steps=1,
        config_file=None,
    ):

        super().__init__(env_dir, use_gui, num_seconds, start_at, action_every_steps)

        # Defining used modules
        self.cells = CellsModule(
            self._cell_shapes,
            self._edges_in_cells,
            self.cell_max_height,
            self.cell_max_width,
            self.cell_height,
            self.cell_width,
        )
        self.emissions = EmissionsModule(
            self.cells, self.output_dir, [EmissionType.NOx], action_every_steps, False
        )

        self.set_extra_modules([self.cells, self.emissions])

        self.obs_shape = (self.cells.yCount, self.cells.xCount, 3)

    def step_actions(self, actions):
        # print(actions)
        for cell_id in actions:
            action = actions[cell_id]
            if action == 1:
                self.cells.close_cell(cell_id)
            else:
                self.cells.open_cell(cell_id)

    def compute_observations(self):
        current_obs = {}
        for cell_id in self.cells.cells_to_edges:

            current_cell = self.cells.cells[cell_id]

            board = np.zeros(self.obs_shape)

            # Get emissions
            board[:, :, 0] = (
                self.emissions.get_emissions_type_matrix(EmissionType.NOx) / 200
            )

            # Get number of vehicles
            for edge in self.cells.edge_to_cells:
                carCount = self.traci.edge.getLastStepVehicleNumber(edge)
                for edge_cell_id in self.cells.edge_to_cells[edge]:
                    cell_obj = self.cells.cells[edge_cell_id]
                    board[cell_obj.matrixPosY, cell_obj.matrixPosX, 0] += carCount

            board[:, :, 0] /= 200
            board[board[:, :, 0] > 1.0] = 1.0
            # Set closed cells
            for closed_cell_id in self.cells.closed_cells:
                cell_obj = self.cells.cells[closed_cell_id]
                board[cell_obj.matrixPosY, cell_obj.matrixPosX, 1] = 1.0

            # Set current looking cell
            board[current_cell.matrixPosY, current_cell.matrixPosX, 2] = 10.0
            current_action_timestep = (
                int((self.sim_step - self.start_at) / self.action_every_steps) - 1
            )
            timestamp_one_hot = np.zeros(shape=(96,), dtype=np.uint8)
            timestamp_one_hot[current_action_timestep] = 1.0
            action_mask = np.ones(shape=(2,), dtype=np.uint8)
            extra = np.array(timestamp_one_hot)
            obs = {"obs": board, "action_mask": action_mask, "extra": extra}
            current_obs[cell_id] = obs
        return current_obs

    def compute_rewards(self):
        closed_percentage = len(self.cells.closed_cells) / len(self.cells.cells_to_edges)

        rewards = {}
        cell_emissions = []
        for cell_id in self.cells.cells_to_edges:
            cell = self.cells.cells[cell_id]

            cell_emission = self.emissions.get_cell_emissions(cell_id, EmissionType.NOx)
            rewards[cell_id] = (
                (100 - cell_emission) / 200
            )
        return rewards

    def _reset(self):
        pass
