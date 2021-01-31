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

    def __init__(self, architecture: str = 'simulation', name: str = 'database'):
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
        self.name = name  # name of the database

    def add_data(self):
        """
        Add a set of data to the database.

        Returns
        -------

        """
        pass

    def _resize_dataset(self, structure: dict):
        """
        Resize a dataset so more data can be added

        Parameters
        ----------
        structure : dict
                path to the dataset that needs to be resized. e.g. {'Na': {'velocities' 100}} will resize all 'x', 'y',
                and 'z' datasets by 100 entries.

        Returns
        -------

        """
        # ensure the database already exists
        try:
            database = hf.File(self.name, 'r+')
        except DatabaseDoesNotExist:
            raise DatabaseDoesNotExist

        # construct the architecture dict
        architecture = self._build_path_input(structure=structure)

        for identifier in architecture:
            for data in database[identifier]:
                data.resize(architecture[identifier], 1)

    def _initialize_database(self, structure: dict):
        """
        Build a database with a general structure.

        Note, this method WILL overwrite a pre-existing database. This is because it is only to be called on the initial
        construction of an experiment class and the first addition of data to it.

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

        database = hf.File(self.name, 'w')  # open the database
        for item in structure:
            self._add_group_structure(item, database)  # add the groups to the database
            self._add_dataset(item, database)          # add a dataset to the groups

    def _build_path_input(self, structure: dict):
        """
        Build an input to a hdf5 database from a dictionary

        In many cases, whilst a dict can be passed on to a method, it is not ideal for use in the hdf5 database. This
        method takes a dictionary and return a new dictionary with the relevant file path.

        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes.
                e.g. {'Na': {'Forces': (200, 5000, 3)}, 'Pressure': (5000, 6), 'Temperature': (5000, 1)}
                In this case, the last value in the tuple corresponds to the number of components that wil be parsed
                to the database. The value in the dict can also be an integer corresponding to a resize operation such
                as {'Na': {'velocities' 100}}. In any case, the deepest portion of the dict must be a non-dict object
                and will be returned as the value of the path to it in the new dictionary.

        Returns
        -------
        architecture : dict
                Corrected path in the hdf5 database. e.g. {'/Na/Velocities': 100}, or {'/Na/Forces': (200, 5000, 3)}

        Examples
        --------
        >>> self._build_path_input(structure = {'Na': {'Forces': (200, 5000, 3)})
        {'Na/Forces': (200, 5000, 3)}
        >>> self._build_path_input(structure={'Na': {'velocities' 100}})
        {'Na/Velocities': 100}
        """

        # Build file paths for the addition.
        architecture = {}
        for group in structure:
            if type(structure[group]) is not dict:
                architecture[group] = structure[group]
            else:
                for subgroup in structure[group]:
                    db_path = os.path.join(group, subgroup)
                    architecture[db_path] = structure[group][subgroup]

        return architecture

    def _add_dataset(self, structure: dict, database: hf.File):
        """
        Add a dataset of the necessary size to the database

        Just as a separate method exists for building the group structure of the hdf5 database, so too do we include
        a separate method for adding a dataset. This is so datasets can be added not just upon the initial construction
        of the database, but also if data is added in the future that should also be stored. This method will assume
        that a group has already been built, although this is not necessary for HDF5, the separation of the actions is
        good practice.

        Parameters
        ----------
        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}
        database: hf.File
                Open hdf5 database object to be added to.

        Returns
        -------
        Updates the database directly.
        """

        architecture = self._build_path_input(structure)  # get the correct file path
        dataset_information = list(architecture.values())[0]  # get the tuple information
        dataset_path = list(architecture)[0]  # get the dataset path in the database

        # Check for a type error in the dataset information
        try:
            if type(dataset_information) is not tuple:
                print("Invalid input for dataset generation")
                raise TypeError
        except TypeError:
            raise TypeError

        # get the correct maximum shape for the dataset -- changes if a system property or an atomic property
        if len(dataset_information[:-1]) == 1:
            max_shape = (None,)
        else:
            max_shape = (dataset_information[0], None)

        for i in range(dataset_information[-1]):
            database[dataset_path].create_dataset(str(i),
                                                  dataset_information[:-1],
                                                  maxshape=max_shape,
                                                  scaleoffset=5)

    def _add_group_structure(self, structure: dict, database: hf.File):
        """
        Add a simple group structure to a database.

        This method will take an input structure and build the required group structure in the hdf5 database. This will
        NOT however instantiate the dataset for the structure, only the group hierarchy.

        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}
        database: hf.File
                Open hdf5 database object to be added to.

        Returns
        -------
        Updates the database directly.
        """

        # Build file paths for the addition.
        architecture = self._build_path_input(structure=structure)

        if list(architecture)[0] in database:
            print("Group structure already exists")
            return
        else:
            database.create_group(list(architecture)[0])

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
