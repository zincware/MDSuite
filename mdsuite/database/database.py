"""
Class for database objects and all of their operations
"""

import h5py as hf
import os

from mdsuite.utils.exceptions import *


class Database:
    """
    Database class

    Databases make up a large part of the functionality of MDSuite and are kept fairly consistent in structure.
    Therefore, the database structure we are using has a separate class with commonly used methods which act as
    wrappers for the hdf5 database.

    Attributes
    ----------
    architecture : str
                The type of the database implemented, either a simulation database, or an analysis database.
    name : str
            The name of the database in question.
    """

    def __init__(self, architecture='simulation', name='database'):
        """
        Constructor for the database class.

        Parameters
        ----------
        architecture : str
                The type of the database implemented, either a simulation database, or an analysis database.
        name : str
                The name of the database in question.
        """

        self.architecture = architecture  # architecture of database
        self.name = name                  # name of the database

    def add_data(self):
        """
        Add a set of data to the database.

        Returns
        -------

        """
        pass

    def _resize_dataset(self, dataset: dict):
        """
        Resize a dataset so more data can be added

        Parameters
        ----------
        dataset : dict
                path to the dataset that needs to be resized. e.g. {'Na': 'velocities'} will resize all 'x', 'y', and
                'z' datasets.

        Returns
        -------

        """
        # ensure the database already exists
        try:
            database = hf.File(self.name, 'r+')
        except DatabaseDoesNotExist:
            raise DatabaseDoesNotExist

        # construct the operations dict
        operations = {}

        for group in dataset:
            if isinstance(dataset[group], int):
                operations[group] = dataset[group]
            else:
                for subgroup in dataset[group]:
                    db_path = os.path.join(group, subgroup)
                    operations[db_path] = dataset[group][subgroup]

        for identifier in operations:
            for data in database[identifier]:
                data.resize(operations[identifier], 1)

    def _initialize_database(self, structure: dict):
        """
        Build a database with a general structure

        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes.
                e.g. {'Na': {'Forces': (200, 5000, 3)}, 'Pressure': (5000, 6), 'Temperature': (5000, 1)}
                In this case, the last value in the tuple corresponds to the number of components that wil be parsed
                to the database.
        Returns
        -------

        """
        for item in structure:

    def _add_group_structure(self, structure: dict):
        """
        Add a simple group structure to a database.

        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}

        Returns
        -------

        """
        # check for parent group existence
        # add groups and datasets

    def _get_memory_information(self, groups=None):
        """
        Get memory information from the database

        Parameters
        ----------
        groups : dict
                Different groups to look at, if set to None, all the group data will be returned. Values of the keys
                correspond to datasets within a group, if they are not given, all datasets will be looked at and
                returned.
        Returns
        -------

        """
        pass
