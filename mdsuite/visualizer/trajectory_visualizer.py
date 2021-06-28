"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Run a trajectory visualization on the atoms or molecules.
"""
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl import MeshData
import importlib.resources
from mdsuite.database.simulation_database import Database
import os
import numpy as np
import tensorflow as tf
import json
from PIL import ImageColor
from mdsuite.utils.meta_functions import join_path


class TrajectoryVisualizer:
    """
    Perform a trajectory visualization using PyQT and openGL.

    Attributes
    ----------
    experiment : object
                Experiment object from which the visualizer will gather information.
    species : list
            A list of species to visualize.
    molecules : list
            If true, molecules will be visualized.
    app : pg.mkQApp
            PyQT application.
    view : gl.GLViewWidget
            View window for openGL.

    Notes
    -----
    TODO: Implement memory safe visualization.
    """

    def __init__(self, experiment: object, species: list = None, molecules: bool = False, unwrapped: bool = False):
        """
        Constructor for the Trajectory Visualizer.

        Parameters
        ----------
        experiment : object
                Experiment object from which the visualizer will gather information.
        species : list
                A list of species to visualize.
        molecules : list
                If true, molecules will be visualized.

        Notes
        -----
        add colour and radius information.
        """
        self.experiment = experiment
        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"))
        self.identifier = "Positions"
        self.species = species
        self.molecules = molecules
        self.unwrapped = unwrapped
        self._check_input()
        self._fill_species_properties()

        self.app = None
        self.view = None

        # Load the data into memory.
        for item in self.species:
            self.data[item]['positions'] = self.database.load_data(path_list=[join_path(item, self.identifier)],
                                                                   select_slice=np.s_[:])
            self.data[item]['mesh'] = MeshData.sphere(20, 20, radius=self.data[item]['mass'])
            self.data[item]['spheres'] = []
            colours = np.random.rand(self.data[item]['mesh'].faceCount(), 4)
            colours[:, 0] = self.data[item]['colour'][0]
            colours[:, 1] = self.data[item]['colour'][1]
            colours[:, 2] = self.data[item]['colour'][2]
            colours[:, 3] = 1
            self.data[item]['mesh'].setFaceColors(colours)

            for _ in range(len(self.data[item]['positions'])):
                object = gl.GLMeshItem(meshdata=self.data[item]['mesh'],
                                       smooth=False,
                                       shader='shaded',
                                       glOptions='opaque')
                object.setMeshData(meshdata=self.data[item]['mesh'])
                self.data[item]['spheres'].append(object)

    def _fill_species_properties(self):
        """
        Add colours and sizes to species.

        Returns
        -------
        Updates the class attributes.
        """
        with importlib.resources.open_text("mdsuite.data", 'PubChemElements_all.json') as json_file:
            pse = json.loads(json_file.read())

        # Try to get the species tensor_values from the Periodic System of Elements file
        for element in self.species:
            for entry in pse:
                if pse[entry][1] == element:
                    self.data[element]['colour'] = list(ImageColor.getcolor(f'#{pse[entry][4]}', 'RGB'))
                    self.data[element]['mass'] = float(pse[entry][3]) / 15

            # TODO Check for no data input.

    def _check_input(self):
        """
        Check the input of the class and update attributes to defaults if necessary.

        Returns
        -------
        Will update the class state if necessary.
        """
        if self.molecules:
            self.species = list(self.experiment.molecules)
        if self.species is None:
            self.species = list(self.experiment.species)
        if self.unwrapped:
            self.identifier = 'Unwrapped_Positions'

        self.data = {}
        for item in self.species:
            self.data[item] = {}

    def _start_app(self):
        """
        Initialize the application and its window structure.

        Returns
        -------

        """
        self.app = pg.mkQApp('MDSuite Visualizer')
        self.view = gl.GLViewWidget()
        self.view.show()

    def _load_starting_configuration(self):
        """
        Initialize the atoms in the window.

        Returns
        -------
        Updates the app window.
        """
        for item in self.data:
            for i, sphere in enumerate(self.data[item]['spheres']):
                posx = self.data[item]['positions'][i][0][0]
                posy = self.data[item]['positions'][i][0][1]
                posz = self.data[item]['positions'][i][0][2]
                sphere.translate(posx, posy, posz)
                self.view.addItem(sphere)

    def run(self):
        """
        Run the visualization.

        Returns
        -------
        Spawns the QT window and runs.
        """
        self._start_app()  # begin running the app
        self._load_starting_configuration()  # load the initial particle positions.

        self.app.exec_()


if __name__ == '__main__':
    """
    Run code tests.
    """
    tst = TrajectoryVisualizer()
    tst.run()
