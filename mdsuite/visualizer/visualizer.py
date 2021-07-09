"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Visualize a simulation.
"""
from mdsuite.database.simulation_database import Database
import open3d as o3d
from os.path import join
import threading


class SimulationVisualizer:
    """
    Visualize a simulation.
    """

    def __init__(self,
                 experiment: object,
                 species: list = None,
                 molecules: bool = False,
                 unwrapped: bool = False):
        """
        Constructor for the visualizer.

        Parameters
        ----------
        experiment : object
                 Experiment object from which the visualizer will gather
                 information.
        species : list
                A list of species to visualize.
        molecules : list
                If true, molecules will be visualized.
        """
        # Particle information
        self.experiment = experiment
        #self.database = Database(name=join(self.experiment.database_path,
        #                                   "database.hdf5"))
        self.species = species
        self.molecules = molecules
        if unwrapped:
            self.identifier = "Unwrapped_Positions"
        else:
            self.identifier = "Positions"

        # App information
        self.is_done = False
        self.main_vis = None
        self.snapshot_pos = None
        self.n_snapshots = 0

    def _on_snapshot(self):
        """
        Commands to run when the snapshot button is pressed.

        Returns
        -------
        Take a snapshot image in a new window.
        """
        pass

    def _on_run_simulation(self):
        """
        Commands to run when the simulation button is pressed.

        Returns
        -------
        Run the simulation in the window.
        """
        pass

    def _on_main_window_closing(self):
        """
        Commands to run when the main window is closed.

        Returns
        -------
        Sets the is_done attribute to True.
        """
        self.is_done = True

        return True

    def _instantiate_window(self):
        """
        Instantiate the actual visualizer window.

        Returns
        -------

        """
        self.main_vis = o3d.visualization.O3DVisualizer(
            "MDSuite"
        )

        self.main_vis.add_action("Snapshot",
                                 self._on_snapshot)
        self.main_vis.add_action("Run Simulation",
                                 self._on_run_simulation)
        self.main_vis.set_on_close(self._on_main_window_closing)

    def _add_atoms(self):
        """
        Add the initial atoms to the system.
        Returns
        -------

        """

    def _update_thread(self):
        """
        Update the thread.

        Returns
        -------

        """
        pass

    def run_app(self):
        """
        Run the app.

        Returns
        -------
        Launches the app.
        """
        # define the app and initialize it.
        app = o3d.visualization.gui.Application.instance
        app.initialize()

        # Define the window and toolbars.
        self._instantiate_window()

        app.add_window(self.main_vis)  # add the window to the app.

        # Start the controls thread.
        threading.Thread(target=self._update_thread).start()

        app.run()

