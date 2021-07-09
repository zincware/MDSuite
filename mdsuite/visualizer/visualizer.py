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
import tensorflow as tf
import threading
from PIL.ImageColor import getcolor
import importlib.resources
import json
import numpy as np
from mdsuite.utils.meta_functions import join_path


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
        self.database = Database(name=join(self.experiment.database_path,
                                           "database.hdf5"))
        self.molecules = molecules
        self.species = species
        if unwrapped:
            self.identifier = "Unwrapped_Positions"
        else:
            self.identifier = "Positions"
        self._check_input()  # check inputs and build data structure.
        self._fill_species_properties()

        # App information
        self.is_done = False
        self.main_vis = None
        self.snapshot_pos = None
        self.n_snapshots = 0
        self.mesh_dict = {}

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
        # Loop over configurations
        # Update current configuration
        # get distances
        # apply translation
        # Update image.
        for i in range(self.experiment.number_of_configurations):
            new_pos = tf.concat([self.data[item]['differences'][:, i] for item
                                 in self.species], axis=0)
            for j, particle in enumerate(self.mesh_dict):
                self.mesh_dict[particle].translate(new_pos[j])
                self.main_vis.remove_geometry(particle)
                self.main_vis.add_geometry(particle,
                                           self.mesh_dict[particle])

            self.main_vis.post_redraw()

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

        self.main_vis.add_action("Run Simulation",
                                 self._on_run_simulation)
        self.main_vis.set_on_close(self._on_main_window_closing)

    def _build_particles(self):
        """
        Add the initial atoms to the system.
        Returns
        -------

        """
        counter = 0
        for item in self.species:
            # load all the data into the dict.
            data = self.database.load_data(
                path_list=[join_path(item, self.identifier)],
                select_slice=np.s_[:])
            self.data[item]['positions'] = data[:, 0]
            self.data[item]['differences'] = tf.experimental.numpy.diff(
                data, axis=1
            )

            colour = list(np.round(np.array(self.data[item]['colour']) / 255,
                                   1))

            for atom in self.data[item]['positions']:
                mesh = o3d.geometry.TriangleMesh.create_sphere(
                    radius=self.data[item]['mass'],
                    resolution=5)
                mesh.translate(atom)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(colour)
                self.mesh_dict[f'particle_{counter}'] = mesh
                self.main_vis.add_geometry(f'particle_{counter}', mesh)
                counter += 1
        self.main_vis.reset_camera_to_default()  # set camera (is it needed?)

    def _update_thread(self):
        """
        This method can interactively update the display of the particles
        but is not used for button events. Currently the only thing it can do
        is add particles to the scene.

        Returns
        -------

        """
        # add the initial atoms to the visualizer.
        o3d.visualization.gui.Application.instance.post_to_main_thread(
            self.main_vis, self._build_particles
        )

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
        self.snapshot_pos = (self.main_vis.os_frame.x,
                             self.main_vis.os_frame.y)

        # Start the controls thread.
        threading.Thread(target=self._update_thread).start()

        app.run()
