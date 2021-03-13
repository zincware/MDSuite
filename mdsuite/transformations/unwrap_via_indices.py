"""
Unwrap a set of coordinates based on dumped indices.
"""

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database
from mdsuite.utils.meta_functions import join_path
import os
import sys


class UnwrapViaIndices(Transformations):
    """ Class to unwrap coordinates based on dumped index values """

    def __init__(self, experiment: object, species: list = None):
        """
        Constructor for the Ionic current calculator.

        Parameters
        ----------
        experiment : object
                Experiment this transformation is attached to.
        species : list
                Species on which this transformation should be applied.
        """
        super().__init__()
        self.experiment = experiment
        self.species = species

        self.database = Database(name=os.path.join(self.experiment.database_path, "database.hdf5"),
                                 architecture='simulation')

        if self.species is None:
            self.species = list(self.experiment.species)

    def _check_for_indices(self):
        """
        Check the database for indices

        Returns
        -------

        """
        truth_table = []
        for item in self.species:
            path = join_path(item, 'Box_Images')
            truth_table.append(self.database.check_existence(path))

        if not all(truth_table):
            print("Indices were not included in the database generation. Please check our simulation files.")
            sys.exit(1)

    def _unwrap_particles(self):
        """
        Perform the unwrapping
        Returns
        -------
        Updates the database object.
        """
        # Loop over batches
        # Apply transformation to batch
        # update database

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self._check_for_indices()
        self._unwrap_particles()  # run the transformation
