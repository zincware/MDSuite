"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import importlib.resources
import json

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
from PIL.ImageColor import getcolor

from mdsuite.database.simulation_database import Database
from mdsuite.utils.meta_functions import join_path

import time
import threading


class SimulationVisualizer:
    """
    Visualize a simulation.
    """

    def __init__(
        self,
        experiment: object,
        species: list = None,
        molecules: bool = False,
        unwrapped: bool = False,
        number_of_configurations: int = None,
        frame_rate: float = 24
    ):
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
        unwrapped : bool
                If true, unwrapped coordinates are studied.
        frame_rate : float
                Frame rate at which to run the visualization in frames per second.
        """
        self.counter = 0
        # Particle information
        self.experiment = experiment
        self.database = Database(
            name=join_path(self.experiment.database_path, "database.hdf5")
        )
        self.frame_rate = frame_rate
        self.molecules = molecules
        self.species = species
        self.number_of_configurations = number_of_configurations
        if unwrapped:
            self.identifier = "Unwrapped_Positions"
        else:
            self.identifier = "Positions"
        self._check_input()  # check inputs and build data structure.
        self._fill_species_properties()
        self._build_atoms_list()

        # App attributes
        self.app = None
        self.vis = None
        self._build_app()

    def _fill_species_properties(self):
        """
        Add colours and sizes to species.

        Returns
        -------
        Updates the class attributes.
        """
        data_name = "mdsuite.data"
        json_name = "PubChemElements_all.json"
        with importlib.resources.open_text(data_name, json_name) as json_file:
            pse = json.loads(json_file.read())

        # Try to get the species tensor_values
        for element in self.species:
            for entry in pse:
                if pse[entry][1] == element:
                    self.data[element]["colour"] = getcolor(f"#{pse[entry][4]}", "RGB")
                    self.data[element]["mass"] = float(pse[entry][3]) / 25
                    self.data[element]["particles"] = self.database.load_data(
                        path_list=[join_path(element, self.identifier)],
                        select_slice=np.s_[:],
                    )[join_path(element, self.identifier)]

    def _check_input(self):
        """
        Check the input of the class and update attributes to defaults if
        necessary.

        Returns
        -------
        Will update the class state if necessary.
        """
        self.data = {}
        for item in self.species:
            self.data[item] = {}

    def _mesh(self, location: np.ndarray, radius: float, colour: np.ndarray):
        """
        Return a mesh object coloured by element.
        """
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
        mesh.compute_vertex_normals()
        mesh.translate(location)
        mesh.paint_uniform_color(colour)

        return mesh

    def _build_particles(self, configuration: int):
        """
        Add the initial atoms to the system.

        Parameters
        ----------
        configuration : int
                Configuration number to load.
        Returns
        -------
        """
        output = []
        for element in self.species:
            colour = np.array(self.data[element]["colour"]) / 255
            radius = self.data[element]["mass"]
            for atom in self.data[element]["particles"]:
                output.append(
                    self._mesh(
                        location=atom[configuration], colour=colour, radius=radius
                    )
                )

        return output

    def _update_position(self, configuration: int):
        """
        Update the position of a particle.

        Parameters
        ----------
        configuration : int
                Configuration to update to.

        Returns
        -------

        """
        output = []
        i = 0
        for element in self.species:
            for atom in self.data[element]["particles"]:
                self.atoms_list[i].translate(atom[configuration], relative=False)
                i += 1

    def _build_atoms_list(self):
        """
        Build the simulation array.
        Returns
        -------
        Updates the class state.
        """
        self.atoms_list = self._build_particles(0)

    def _build_app(self):
        """
        Build the app window and update the class.

        Returns
        -------
        Updates the class.
        """
        self.app = gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer("MDSuite Visualizer", 1024, 768)
        self.vis.show_settings = True
        # Add meshes
        self.vis.reset_camera_to_default()

        self.vis.add_action("Run Simulation", self._continuous_trajectory)

        self.app.add_window(self.vis)

    def _continuous_trajectory(self, vis):
        """
        Button command for running the simulation in the visualizer.
        Parameters
        ----------
        vis : visualizer
                Object passed during the callback.
        """
        self.counter = 0
        threading.Thread(target=self._run_trajectory).start()

    def _run_trajectory(self):
        """
        Callback method for running the trajectory smoothly.
        Returns
        -------
        Runs through the trajectory.
        """
        for step in range(self.number_of_configurations):
            time.sleep(1 / self.frame_rate)
            o3d.visualization.gui.Application.instance.post_to_main_thread(
                self.vis, self._updated_scene
            )

    def _updated_scene(self, configuration: int = None, seed: bool = False):
        """
        Update the scene.

        Parameters
        ----------
        configuration : int (default = True)
                Configuration to load.
        seed : bool (default = False)
                If true, no geometries are removed before being added.
        Returns
        -------
        Updates the positions of the particles in the app.
        """
        if configuration is None:
            if self.counter + 1 == self.number_of_configurations:
                self.counter = 0

            if not seed:
                self._update_position(self.counter)

            for i in range(len(self.atoms_list)):
                if not seed:

                    self.vis.remove_geometry(f"sphere_{i}")
                self.vis.add_geometry(f"sphere_{i}", self.atoms_list[i])
            self.counter += 1
        else:
            if configuration > self.number_of_configurations - 1:
                configuration = self.number_of_configurations - 1
            if not seed:
                self._update_position(configuration)
            for i in range(len(self.atoms_list)):
                if not seed:
                    self.vis.remove_geometry(f"sphere_{i}")
                self.vis.add_geometry(f"sphere_{i}", self.atoms_list[i])

    def run_app(self, starting_configuration: int = None):
        """
        Run the app.

        Parameters
        ----------
        starting_configuration : int
                Starting configuration for the visualization.

        Returns
        -------
        Launches the app.
        """
        self._updated_scene(configuration=starting_configuration, seed=True)
        self.vis.reset_camera_to_default()

        self.app.run()
