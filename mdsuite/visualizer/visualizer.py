"""
Python module to visualize tensor_values in the mds database structure.

Notes
-----
This is an experimental module. It is envisaged that in the future, this visualization will include the 2d and 3d
plotting, as well as approaches for visualizing vector fields.
"""
import vpython as vp
import os
import numpy as np
import random
import importlib.resources
import json
import colorutils as cu

from mdsuite.database.simulation_database import Database
from mdsuite.utils.meta_functions import join_path


class TrajectoryVisualizer:
    """
    Class for visualizing trajectories of atoms.
    """

    def __init__(self, experiment: object, species: list, unwrapped: bool = False):
        """
        Constructor for the visualizer class.

        Parameters
        ----------
        experiment : object
                Experiment class that will call the visualizer.
        species: list
                Species you wish to visualize
        unwrapped : bool
                If true, unwrapped positions will be visualized.
        """
        self.experiment = experiment
        self.species = species
        self.database = Database(name=os.path.join(self.experiment.database_path, 'database.hdf5'))
        self.data_dictionary = {}
        self.canvas: vp.canvas

        if unwrapped:
            self.loaded_property = 'Unwrapped_Positions'
        else:
            self.loaded_property = 'Positions'
        self._populate_data_dict()

    @staticmethod
    def _get_hue() -> float:
        """
        Get a hue value for use in the colour generation.

        Returns
        -------
        hue : float
                hue value for an rsv colour designed for optimal spread of colour choice.
        """
        grc = 0.618033988749895
        h = random.uniform(0.1, 1.0) + grc

        return h % 1

    def _populate_data_dict(self):
        """
        add species, colour, and size information to the tensor_values dict.
        Returns
        -------
        updates the self.data_dict class state
        """
        with importlib.resources.open_text("mdsuite.data", 'PubChemElements_all.json') as json_file:
            pse = json.loads(json_file.read())

        for species in self.species:
            colour = None
            mass = None
            for entry in pse:
                if pse[entry][1] == species:
                    colour = cu.Color(hex=pse[entry][4]).hsv
                    mass = float(pse[entry][3]) / 30

            if colour is None:
                hue = self._get_hue()
                colour = (hue, 0.25, 0.8)
            if mass is None:
                mass = round(random.uniform(1.0, 2.0), 3)

            self.data_dictionary[species] = {'radius': mass,
                                             'colour': vp.vector(colour[0], colour[1], colour[2]),
                                             'spheres': []}

    def _load_trajectory(self):
        """
        Load the full trajectory of each species from the database.

        Returns
        -------
        trajectory: dict
                Dictionary of species and positions for use in the visualization

        Notes
        -----
        This whole class is currently NOT memory safe. This is because working out the best way to pre-load trajectory
        states at the right time is challenging and still a work in progress. Therefore, this will load the full
        trajectory into memory.
        """
        for species in self.species:
            path_list = join_path(species, self.loaded_property)
            self.data_dictionary[species]['positions'] = self.database.load_data(path_list=[path_list],
                                                                                 select_slice=np.s_[:])

    def _slider(self):
        """
        Build the slider object.

        Returns
        -------
        updates the visualizer
        """
        def S(s):
            """
            Operate the slider
            Parameters
            ----------
            s : int
            Returns
            -------

            """
            self._update_positions(s.value)
        vp.slider(min=0, max=int(self.experiment.number_of_configurations - 1), step=1, bind=S)

    def _run_radio(self):
        """
        Build the run button.
        Returns
        -------
        Updates the visualiser
        """
        def R(r):
            """
            Operate the run button
            Parameters
            ----------
            r : object
            Returns
            -------
            updates the button
            """
            if r.checked:
                self._run_full_trajectory()
            if not r.checked:
                self._update_positions(0)

        vp.radio(bind=R, text='Run')

    def _set_scene(self):
        """
        set box properties for atom loading.

        Returns
        -------

        """
        self.canvas = vp.canvas(background=vp.color.hsv_to_rgb(vp.vector(197, 0.03, 0.88)), visible=False)
        self.canvas.append_to_caption('\n\n')
        self._run_radio()
        self.canvas.append_to_caption('\n\n')
        self._slider()
        self.canvas.append_to_caption('\n\n')

    def _draw_spheres(self):
        """
        Draw the spheres in the configuration for the first time step.

        Returns
        -------

        """
        for species in self.data_dictionary:
            for atom in self.data_dictionary[species]['positions']:
                self.data_dictionary[species]['spheres'].append(vp.sphere(pos=vp.vector(atom[0][0],
                                                                                        atom[0][1],
                                                                                        atom[0][2]),
                                                                          radius=self.data_dictionary[species][
                                                                              'radius'],
                                                                          color=vp.color.hsv_to_rgb(
                                                                              self.data_dictionary[species]['colour'])))

    def _update_positions(self, counter: int):
        """
        Update the positions of the spheres.

        Parameters
        ----------
        counter : int
                Time step we are up to in the simulation.
        Returns
        -------
        updates the class state
        """
        for species in self.data_dictionary:
            for i, atom in enumerate(self.data_dictionary[species]['positions']):
                self.data_dictionary[species]['spheres'][i].pos = vp.vector(atom[counter][0],
                                                                            atom[counter][1],
                                                                            atom[counter][2])

    def _run_full_trajectory(self):
        """
        Loop over the full trajectory
        """
        for time_step in range(1, int(self.experiment.number_of_configurations)):
            self._update_positions(counter=time_step)
            vp.rate(100)

    def run_visualization(self):
        """
        Run the visualization.

        Returns
        -------

        """
        self._load_trajectory()
        self._set_scene()
        self._draw_spheres()
        self.canvas.visible = True
