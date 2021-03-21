"""
Python module to calculate the ionic current in a experiment.
"""

import numpy as np
from tqdm import tqdm
import os
import tensorflow as tf

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database
from mdsuite.utils.meta_functions import join_path
from mdsuite.database.data_manager import DataManager
from mdsuite.memory_management.memory_manager import MemoryManager


class ThermalFlux(Transformations):
    """
    Class to generate and store the ionic current of a experiment

    Attributes
    ----------
    experiment : object
            Experiment this transformation is attached to.
    """

    def __init__(self, experiment: object):
        """
        Constructor for the Ionic current calculator.

        Parameters
        ----------
        experiment : object
                Experiment this transformation is attached to.
        calculator : Calculator
        """
        super().__init__()
        self.experiment = experiment
        self.batch_size: int
        self.n_batches: int
        self.remainder: int

        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"),
                                 architecture='simulation')

    def _prepare_monitors(self, data_path: list):
        """
        Prepare the tensor_values and memory managers.

        Parameters
        ----------
        data_path : list
                List of tensor_values paths to load from the hdf5 database_path.

        Returns
        -------

        """
        self.memory_manager = MemoryManager(data_path=data_path, database=self.database, scaling_factor=5,
                                            memory_fraction=0.5)
        self.data_manager = DataManager(data_path=data_path, database=self.database)
        self.batch_size, self.n_batches, self.remainder = self.memory_manager.get_batch_size()
        self.data_manager.batch_size = self.batch_size
        self.data_manager.n_batches = self.n_batches
        self.data_manager.remainder = self.remainder

    def _prepare_database_entry(self):
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------

        """
        # collect machine properties and determine batch size
        path = join_path('Thermal_Flux', 'Thermal_Flux')  # name of the new database_path
        dataset_structure = {path: (self.experiment.number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure)  # add a new dataset to the database_path
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return data_structure

    def _save_coordinates(self, data: tf.Tensor, index: int, batch_size: int, data_structure: dict):
        """
        Save the tensor_values into the database_path

        Parameters
        ----------
        species : str
                name of species to update in the database_path.
        Returns
        -------
        saves the tensor_values to the database_path.
        """
        self.database.add_data(data=data,
                               structure=data_structure,
                               start_index=index,
                               batch_size=batch_size,
                               system_tensor=True)

    def _transformation(self, data: tf.Tensor):
        """
        Compute the ionic current of the experiment.

        Parameters
        ----------
        batch_number
        remainder
        Returns
        -------
        system_current : np.array
                System current as a numpy array.
        """

        phi_x = np.multiply(data[1][:, :, 0], data[0][:, :, 0]) + \
                np.multiply(data[1][:, :, 3], data[0][:, :, 1]) + \
                np.multiply(data[1][:, :, 4], data[0][:, :, 2])
        phi_y = np.multiply(data[1][:, :, 3], data[0][:, :, 0]) + \
                np.multiply(data[1][:, :, 1], data[0][:, :, 1]) + \
                np.multiply(data[1][:, :, 5], data[0][:, :, 2])
        phi_z = np.multiply(data[1][:, :, 4], data[0][:, :, 0]) + \
                np.multiply(data[1][:, :, 5], data[0][:, :, 1]) + \
                np.multiply(data[1][:, :, 2], data[0][:, :, 2])

        phi = np.dstack([phi_x, phi_y, phi_z])

        phi_sum = phi.sum(axis=0)
        phi_sum_atoms = phi_sum / self.experiment.units['NkTV2p']  # factor for units lammps nktv2p

        energy = data[3] + data[2]

        energy_velocity = energy * data[0]
        energy_velocity_atoms = energy_velocity.sum(axis=0)
        system_current = energy_velocity_atoms - phi_sum_atoms

        return system_current

    def _compute_thermal_flux(self):
        """
        Loop over the batches, run calculations and update the database_path.
        Returns
        -------
        Updates the database_path.
        """

        data_structure = self._prepare_database_entry()

        species_path = [join_path(species, 'Velocities') for species in self.experiment.species]
        stress_path = [join_path(species, 'Stress') for species in self.experiment.species]
        pe_path = [join_path(species, 'PE') for species in self.experiment.species]
        ke_path = [join_path(species, 'KE') for species in self.experiment.species]
        data_path = np.concatenate((species_path, stress_path, pe_path, ke_path))
        print(data_path)
        self._prepare_monitors(data_path)
        batch_generator, batch_generator_args = self.data_manager.batch_generator(dictionary=True)
        data_set = tf.data.Dataset.from_generator(batch_generator,
                                                  args=batch_generator_args,
                                                  output_types=dict) #,
                                                  #output_signature=tf.TypeSpec(dict))
        data_set.prefetch(tf.data.experimental.AUTOTUNE)
        for index, x in enumerate(data_set):
            data = self._transformation(x)
            self._save_coordinates(data, index, self.batch_size, data_structure)

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        self._compute_thermal_flux()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
