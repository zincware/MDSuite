"""
Python module to calculate the ionic current in a system.
"""

import numpy as np
from tqdm import tqdm
import os

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database
from mdsuite.calculators.calculator import Calculator
from mdsuite.utils.meta_functions import join_path


class IonicCurrent(Transformations):
    """
    Class to generate and store the ionic current of a system

    Attributes
    ----------
    experiment : object
            Experiment this transformation is attached to.
    """

    def __init__(self, experiment: object, calculator: Calculator):
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
        self.calculator = calculator

        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"),
                                 architecture='simulation')

    def _compute_ionic_current(self, batch_number: int = None, remainder: int = None):
        """
        Compute the ionic current of the system.

        Parameters
        ----------
        batch_number
        remainder
        Returns
        -------
        system_current : np.array
                System current as a numpy array.
        """

        velocity_matrix = self.calculator.load_batch(batch_number, loaded_property='Velocities', remainder=remainder)
        # build charge array
        species_charges = [self.experiment.species[atom]['charge'][0] for atom in self.experiment.species]

        system_current = np.zeros((self.calculator.batch_size['Parallel'], 3))  # instantiate the current array
        # Calculate the total system current
        for j in range(len(velocity_matrix)):
            system_current += np.array(np.sum(velocity_matrix[j][:, 0:], axis=0)) * species_charges[j]

        return system_current

    def _prepare_database_entry(self):
        """
        Call some housekeeping methods and prepare for the transformations.
        Returns
        -------

        """
        # collect machine properties and determine batch size
        self.calculator.collect_machine_properties(group_property='Velocities')
        n_batches = np.floor(self.experiment.number_of_configurations / self.calculator.batch_size['Parallel'])
        remainder = int(self.experiment.number_of_configurations % self.calculator.batch_size['Parallel'])
        db_object = self.database.open()  # open a database
        path = join_path('Ionic_Current', 'Ionic_Current')  # name of the new database
        dataset_structure = {path: (self.experiment.number_of_configurations, 3)}
        self.database.add_dataset(dataset_structure, db_object)  # add a new dataset to the database
        data_structure = {path: {'indices': np.s_[:], 'columns': [0, 1, 2]}}

        return n_batches, remainder, data_structure, db_object

    def _run_calculation_loops(self):
        """
        Loop over the batches, run calculations and update the database.
        Returns
        -------
        Updates the database.
        """

        n_batches, remainder, data_structure, db_object = self._prepare_database_entry()
        # process the batches
        for i in tqdm(range(int(n_batches)), ncols=70):
            system_current = self._compute_ionic_current(i)
            self.database.add_data(data=system_current,
                                   structure=data_structure,
                                   database=db_object,
                                   start_index=i,
                                   batch_size=self.calculator.batch_size['Parallel'],
                                   system_tensor=True)

        if remainder > 0:
            start = self.experiment.number_of_configurations - remainder
            system_current = self._compute_ionic_current(remainder=remainder)
            self.database.add_data(data=system_current,
                                   structure=data_structure,
                                   database=db_object,
                                   start_index=start,
                                   batch_size=remainder,
                                   system_tensor=True)

        self.database.close(db_object)  # close the database

    def run_transformation(self):
        """
        Run the ionic current transformation
        Returns
        -------

        """
        self._run_calculation_loops()  # run the transformation.
        self.experiment.memory_requirements = self.database.get_memory_information()
