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
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import time


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

        # Instantiate app attributes
        self.app = None
        self.widget = None
        self.grid = None
        self.trajectory = []
        self._instantiate_window()

    @staticmethod
    def _sphere(location: np.ndarray, colour: np.ndarray, radius: float):
        """
        Return a sphere mesh object.

        Parameters
        ----------
        location : np.ndarray
                Location of the particle.
        colour : np.ndarray
                Colour of the particle. This should be in reduced rgb, i.e.
                (255, 255, 255) = (1, 1, 1).
        radius : float
                Radius of the sphere.
        """
        md = gl.MeshData.sphere(radius=radius, rows=50, cols=50)
        colors = np.ones((md.faceCount(), 4), dtype=float)
        colors[:, 0:3] = colour
        md.setFaceColors(colors)
        m3 = gl.GLMeshItem(meshdata=md,
                           smooth=True,
                           shader='shaded')

        m3.translate(*location)

        return m3

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
        if self.molecules:
            self.species = list(self.experiment.molecules)
        if self.species is None:
            self.species = list(self.experiment.species)

        self.data = {}
        for item in self.species:
            self.data[item] = {}

    def _instantiate_window(self):
        """
        Instantiate the actual visualizer window.

        Returns
        -------

        """
        self._build_app()  # construct the app window and main GL widget.
        self.widget.show()  # make it big at the start.
        #self._build_layout()  # add screen elements.

    def _build_layout(self):
        """
        Build the layout of the window.

        Returns
        -------

        """
        layout = QtGui.QGridLayout()  # instantiate a layout.
        self.widget.setLayout(layout)  # add the layout to the app.

        # Next configuration button.
        button = QtGui.QPushButton("Next configuration")
        button.clicked.connect(self._update_thread)
        layout.addWidget(button, 1, 2)

        # Run simulation button.
        button = QtGui.QPushButton("Run simulation")
        button.clicked.connect(self._run_sim)
        layout.addWidget(button, 1, 1)

    def _build_app(self):
        """
        Build the app window and update the class.

        Returns
        -------
        Updates the class.
        """
        self.app = pg.mkQApp("GLMeshItem Example")
        self.widget = gl.GLViewWidget()
        self.widget.setBackgroundColor(8, 87, 96)
        self.widget.setWindowTitle('MDSuite Visualizer')
        self.widget.setCameraPosition(distance=110)

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
                output.append(self._sphere(location=atom[configuration],
                                           colour=colour,
                                           radius=radius))

        return output

    def _build_simulation(self):
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

    def _update_thread(self, configuration: int = None):
        """
        This method updates the positions of the particles.

        Returns
        -------

        """
        if configuration is None:
            if self.counter + 1 == self.number_of_configurations:
                self.counter = 0
            self.widget.clear()
            self.widget.items = self.trajectory[self.counter]
            self.counter += 1
        else:
            if configuration > self.number_of_configurations - 1:
                configuration = self.number_of_configurations - 1
            self.widget.clear()
            self.widget.items = self.trajectory[configuration]

    def _run_sim(self):
        """
        Run the simulation.

        TODO: Make it work...
        Returns
        -------

        """
        while True:
            self._update_thread()

    def run_app(self):
        """
        Run the app.

        Returns
        -------
        Launches the app.
        """
        self._build_simulation()  # load data into spheres
        self._update_thread()

        self.app.exec()
