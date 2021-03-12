"""
Unwrap a set of coordinates based on dumped indices.
"""

from mdsuite.transformations.transformations import Transformations
from mdsuite.database.database import Database
import os


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

    def run_transformation(self):
        """
        Perform the transformation.
        """
        self.unwrap_particles()  # run the transformation
