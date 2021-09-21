"""
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html.
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Visualize a simulation.
"""
from mdsuite.database.simulation_database import Database
from PIL.ImageColor import getcolor
import importlib.resources
import json
import numpy as np
from mdsuite.utils.meta_functions import join_path
import open3d as o3d
import open3d.visualization.gui as gui


class SimulationVisualizer:
    """
    Visualize a simulation.
    """

    def __init__(self,
                 experiment: object,
                 species: list = None,
                 molecules: bool = False,
                 unwrapped: bool = False,
                 number_of_configurations: int = None):
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
        """
        self.counter = 0
        # Particle information
        self.experiment = experiment
        self.database = Database(name=join_path(
            self.experiment.database_path, "database.hdf5")
        )
        self.molecules = molecules
        self.species = species
        self.number_of_configurations = number_of_configurations
        if unwrapped:
            self.identifier = "Unwrapped_Positions"
        else:
            self.identifier = "Positions"
        self._check_input()  # check inputs and build data structure.
        self._fill_species_properties()
        self._build_trajectory()

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
        json_name = 'PubChemElements_all.json'
        with importlib.resources.open_text(data_name, json_name) as json_file:
            pse = json.loads(json_file.read())

        # Try to get the species tensor_values
        for element in self.species:
            for entry in pse:
                if pse[entry][1] == element:
                    self.data[element]['colour'] = getcolor(
                        f'#{pse[entry][4]}', 'RGB'
                    )
                    self.data[element]['mass'] = float(pse[entry][3]) / 25
                    self.data[element]['particles'] = self.database.load_data(
                        path_list=[join_path(element, self.identifier)],
                        select_slice=np.s_[:]
                    )

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
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius,
                                                       resolution=10)
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
            colour = np.array(self.data[element]['colour']) / 255
            radius = self.data[element]['mass']
            for atom in self.data[element]['particles']:
                output.append(self._mesh(location=atom[configuration],
                                         colour=colour,
                                         radius=radius))

        return output

    def _build_trajectory(self):
        """
        Build the simulation array.
        Returns
        -------
        Updates the class state.
        """
        self.trajectory = [
            self._build_particles(i) for i in range(
                self.number_of_configurations
            )
        ]

    def _build_app(self):
        """
        Build the app window and update the class.

        Returns
        -------
        Updates the class.
        """
        self.app = gui.Application.instance
        self.app.initialize()

        self.vis = o3d.visualization.O3DVisualizer("MDSuite Visualizer",
                                                   1024,
                                                   768)
        self.vis.show_settings = True
        # Add meshes
        self.vis.reset_camera_to_default()

        self.app.add_window(self.vis)

    def _updated_scene(self, configuration: int = None):
        """
        Update the scene.
        Returns
        -------

        """
        if configuration is None:
            if self.counter + 1 == self.number_of_configurations:
                self.counter = 0

            for i in range(len(self.trajectory[self.counter])):
                self.vis.add_geometry(f"sphere_{i}",
                                      self.trajectory[self.counter][i])
            self.counter += 1
        else:
            if configuration > self.number_of_configurations - 1:
                configuration = self.number_of_configurations - 1

            for i in range(len(self.trajectory[configuration])):
                self.vis.add_geometry(f"sphere_{i}",
                                      self.trajectory[configuration][i])

    def run_app(self, starting_configuration: int = None):
        """
        Run the app.

        Returns
        -------
        Launches the app.
        """
        self._updated_scene(configuration=starting_configuration)
        self.vis.reset_camera_to_default()

        self.app.run()
