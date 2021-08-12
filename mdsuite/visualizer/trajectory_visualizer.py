"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
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
        unwrapped : bool
                If true, unwrapped coordinates are studied.
        """
        # Particle information
        self.experiment = experiment
        self.database = Database(name=join_path(self.experiment.database_path,
                                                "database.hdf5"))
        self.molecules = molecules
        self.species = species
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
        self._instantiate_window()

    def _sphere(self):
        """
        Return a sphere mesh object.
        """
        md = gl.MeshData.sphere(rows=10, cols=20)
        colors = np.ones((md.faceCount(), 4), dtype=float)
        colors[::2, 0] = 0
        colors[:, 1] = np.linspace(0, 1, colors.shape[0])
        print(colors)
        md.setFaceColors(colors)
        m3 = gl.GLMeshItem(meshdata=md, smooth=False)  # , shader='balloon')

        m3.translate(5, -5, 0)
        self.widget.addItem(m3)

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
                    self.data[element]['colour'] = getcolor(f'#{pse[entry][4]}',
                                                            'RGB')
                    self.data[element]['mass'] = float(pse[entry][3]) / 15

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

    def _on_run_simulation(self, vis):
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

        return True

    def _instantiate_window(self):
        """
        Instantiate the actual visualizer window.

        Returns
        -------

        """
        self.app = pg.mkQApp("GLMeshItem Example")
        self.widget = gl.GLViewWidget()
        self.widget.show()
        self.widget.setWindowTitle('MDSuite')
        self.widget.setCameraPosition(distance=40)

        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.widget.addItem(self.grid)

    def _build_particles(self):
        """
        Add the initial atoms to the system.
        Returns
        -------

        """
        pass

    def _update_thread(self):
        """
        This method can interactively update the display of the particles
        but is not used for button events. Currently the only thing it can do
        is add particles to the scene.

        Returns
        -------

        """
        # add the initial atoms to the visualizer.
        pass

    def run_app(self):
        """
        Run the app.

        Returns
        -------
        Launches the app.
        """
        # execute the app.
        self._sphere()
        self.app.exec_()
