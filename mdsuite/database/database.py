"""
Class for database_path objects and all of their operations
"""

import h5py as hf
import numpy as np
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.exceptions import *
import tensorflow as tf
import time
from typing import Union


class Database:
    """
    Database class

    Databases make up a large part of the functionality of MDSuite and are kept fairly consistent in structure.
    Therefore, the database_path structure we are using has a separate class with commonly used methods which act as
    wrappers for the hdf5 database_path.

    Attributes
    ----------
    architecture : str
                The type of the database_path implemented, either a simulation database_path, or an analysis database_path.
    name : str
            The name of the database_path in question.
    """

    def __init__(self, architecture: str = 'simulation', name: str = 'database'):
        """
        Constructor for the database_path class.

        Parameters
        ----------
        architecture : str
                The type of the database_path implemented, either a simulation database_path, or an analysis database_path.
        name : str
                The name of the database_path in question.
        """

        self.architecture = architecture  # architecture of database_path
        self.name = name  # name of the database_path

    def open(self, mode: str = 'a') -> hf.File:
        """
        Open the database_path

        Parameters
        ----------
        mode : str
                Mode in which to open the database_path

        Returns
        -------
        database_path : hf.File
                returns a database_path object
        """

        return hf.File(self.name, mode)

    @staticmethod
    def close(database: hf.File):
        """
        Close the database_path

        Parameters
        ----------
        database : hf.File
                Database to close

        Returns
        -------
        Closes the database_path object
        """

        database.close()

    def add_data(self, data: np.array, structure: dict, start_index: int, batch_size: int, tensor: bool = False,
                 system_tensor: bool = False, flux: bool = False, sort: bool = False, n_atoms: int = None):
        """
        Add a set of tensor_values to the database_path.

        Parameters
        ----------
        flux : bool
                If true, the atom dimension is not included in the slicing.
        system_tensor : bool
                If true, no atom information is looked for when saving
        tensor : bool
                If true, this will skip the type enforcement
        batch_size : int
                Number of configurations in each batch
        start_index : int
                Point in database_path from which to start filling.
        database : hf.File
                Database in which to store the tensor_values
        structure : dict
                Structure of the tensor_values to be loaded into the database_path e.g.
                {'Na/Velocities': {'indices': [1, 3, 7, 8, ... ], 'columns' = [3, 4, 5], 'length': 500}}
        data : np.array
                Data to be loaded in.
        sort : bool
                If true, tensor_values is sorted before being dumped into the database_path.
        n_atoms : int
                Necessary if the sort function is called. Total number of atoms in the experiment.
        Returns
        -------
        Adds tensor_values to the database_path
        """

        database = self.open()
        # Loop over items
        stop_index = start_index + batch_size  # get the stop index
        for item in structure:
            if tensor:
                database[item][:, start_index:stop_index, :] = data[:, :, 0:3]
            elif system_tensor:
                database[item][start_index:stop_index, :] = data[:, 0:3]
            elif flux:
                database[item][start_index:stop_index, :] = data[structure[item]['indices']][
                    np.s_[:, structure[item]['columns'][0]:structure[item]['columns'][-1] + 1]].astype(float)
            else:
                database[item][:, start_index:stop_index, :] = self._get_data(data, structure, item, batch_size, sort,
                                                                              n_atoms=n_atoms)
        database.close()

    def _get_data(self, data: np.array, structure: dict, item: str, batch_size: int, sort: bool = False,
                  n_atoms: int = None):
        """
        Fetch tensor_values with some format from a large array.

        Returns
        -------

        """
        if sort:
            indices = self._update_indices(data, structure[item]['indices'], batch_size, n_atoms)
            tf.gather_nd(tf.convert_to_tensor(data), indices)[
                np.s_[:, structure[item]['columns'][0]:structure[item]['columns'][-1] + 1]].astype(float).reshape(
                (structure[item]['length'], batch_size, len(structure[item]['columns'])), order='F')
        else:
            indices = structure[item]['indices']
            return data[indices][
                np.s_[:, structure[item]['columns'][0]:structure[item]['columns'][-1] + 1]].astype(float).reshape(
                (structure[item]['length'], batch_size, len(structure[item]['columns'])), order='F')

    @staticmethod
    def _update_indices(data: np.array, indices: np.array, batch_size: int, n_atoms: int):
        """
        Update the indices key of the structure dictionary if the tensor_values must be sorted.

        Returns
        -------

        """
        atom_ids = np.tile(indices, batch_size)
        simulation_ids = np.split(np.array(data[:, 0]).astype(int), int(batch_size/n_atoms))
        indices = np.zeros(int(batch_size*len(atom_ids)))

        counter = 0
        for i, item in enumerate(simulation_ids):
            stop = counter + len(atom_ids)
            correction = i*n_atoms
            sorter = np.argsort(item)
            simulation_ids[counter:stop] = sorter[np.searchsorted(item, atom_ids, sorter=sorter)] + correction
            counter += len(atom_ids)

        return indices

    def _resize_dataset(self, structure: dict):
        """
        Resize a dataset so more tensor_values can be added

        Parameters
        ----------
        structure : dict
                path to the dataset that needs to be resized. e.g. {'Na': {'velocities' 100}} will resize all 'x', 'y',
                and 'z' datasets by 100 entries.

        Returns
        -------

        """
        # ensure the database_path already exists
        try:
            database = hf.File(self.name, 'r+')
        except DatabaseDoesNotExist:
            raise DatabaseDoesNotExist

        # construct the architecture dict
        architecture = self._build_path_input(structure=structure)

        for identifier in architecture:
            for data in database[identifier]:
                data.resize(architecture[identifier], 1)

    def initialize_database(self, structure: dict):
        """
        Build a database_path with a general structure.

        Note, this method WILL overwrite a pre-existing database_path. This is because it is only to be called on the initial
        construction of an experiment class and the first addition of tensor_values to it.

        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes.
                e.g. {'Na': {'Forces': (200, 5000, 3)}, 'Pressure': (5000, 6), 'Temperature': (5000, 1)}
                In this case, the last value in the tuple corresponds to the number of components that wil be parsed
                to the database_path.
        Returns
        -------

        """

        self.add_dataset(structure)  # add a dataset to the groups

    @staticmethod
    def _build_path_input(structure: dict) -> dict:
        """
        Build an input to a hdf5 database_path from a dictionary

        In many cases, whilst a dict can be passed on to a method, it is not ideal for use in the hdf5 database_path. This
        method takes a dictionary and return a new dictionary with the relevant file path.

        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes.
                e.g. {'Na': {'Forces': (200, 5000, 3)}, 'Pressure': (5000, 6), 'Temperature': (5000, 1)}
                In this case, the last value in the tuple corresponds to the number of components that wil be parsed
                to the database_path. The value in the dict can also be an integer corresponding to a resize operation such
                as {'Na': {'velocities' 100}}. In any case, the deepest portion of the dict must be a non-dict object
                and will be returned as the value of the path to it in the new dictionary.

        Returns
        -------
        architecture : dict
                Corrected path in the hdf5 database_path. e.g. {'/Na/Velocities': 100}, or {'/Na/Forces': (200, 5000, 3)}

        Examples
        --------
        >>> database_path = Database
        >>> database_path._build_path_input(structure = {'Na' : {'Forces': (200, 5000, 3)}})
        {'Na/Forces': (200, 5000, 3)}
        >>> database_path._build_path_input(structure={'Na': {'velocities' 100}})
        {'Na/Velocities': 100}
        """

        # Build file paths for the addition.
        architecture = {}
        for group in structure:
            if type(structure[group]) is not dict:
                architecture[group] = structure[group]
            else:
                for subgroup in structure[group]:
                    db_path = join_path(group, subgroup)
                    architecture[db_path] = structure[group][subgroup]

        return architecture

    def add_dataset(self, structure: dict):
        """
        Add a dataset of the necessary size to the database_path

        Just as a separate method exists for building the group structure of the hdf5 database_path, so too do we include
        a separate method for adding a dataset. This is so datasets can be added not just upon the initial construction
        of the database_path, but also if tensor_values is added in the future that should also be stored. This method will assume
        that a group has already been built, although this is not necessary for HDF5, the separation of the actions is
        good practice.

        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database_path.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}
        database_path: hf.File
                Open hdf5 database_path object to be added to.

        Returns
        -------
        Updates the database_path directly.
        """

        database = self.open()
        architecture = self._build_path_input(structure)  # get the correct file path
        for item in architecture:
            dataset_information = architecture[item]  # get the tuple information
            dataset_path = item  # get the dataset path in the database_path

            # Check for a type error in the dataset information
            try:
                if type(dataset_information) is not tuple:
                    print("Invalid input for dataset generation")
                    raise TypeError
            except TypeError:
                raise TypeError

            # get the correct maximum shape for the dataset -- changes if a experiment property or an atomic property
            if len(dataset_information[:-1]) == 1:
                vector_length = dataset_information[-1]
                max_shape = (None, vector_length)
            else:
                max_shape = list(dataset_information)
                max_shape[1] = None
                max_shape = tuple(max_shape)

            database.create_dataset(dataset_path, dataset_information, maxshape=max_shape, scaleoffset=5)
        database.close()

    def _add_group_structure(self, structure: dict):
        """
        Add a simple group structure to a database_path.

        This method will take an input structure and build the required group structure in the hdf5 database_path. This will
        NOT however instantiate the dataset for the structure, only the group hierarchy.

        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database_path.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}
        database: hf.File
                Open hdf5 database_path object to be added to.

        Returns
        -------
        Updates the database_path directly.
        """

        database = hf.File(self.name)
        # Build file paths for the addition.
        architecture = self._build_path_input(structure=structure)
        for item in list(architecture):
            if item in database:
                print("Group structure already exists")
            else:
                database.create_group(item)
        database.close()

    def get_memory_information(self) -> dict:
        """
        Get memory information from the database_path

        Returns
        -------
        memory_database : dict
                A dictionary of the memory information of the groups in the database_path
        """

        database = hf.File(self.name)

        memory_database = {}
        for item in database:
            for ds in database[item]:
                memory_database[join_path(item, ds)] = database[item][ds].nbytes

        return memory_database

    def check_existence(self, path: str) -> bool:
        """
        Check to see if a dataset is in the database_path

        Parameters
        ----------
        path : str
                Path to the desired dataset

        Returns
        -------
        response : bool
                If true, the path exists, else, it does not.
        """
        database_object = hf.File(self.name, 'r')
        keys = []
        database_object.visit(lambda item: keys.append(database_object[item].name) if type(database_object[item]) is
                                                                                      hf.Dataset else None)

        path = f'/{path}' # add the / to avoid name overlapping
        response = any(list(path in item for item in keys))
        database_object.close()
        return response

    def change_key_names(self, mapping: dict):
        """
        Change the name of database_path keys

        Parameters
        ----------
        mapping : dict
                Mapping for the change of names

        Returns
        -------
        Updates the database_path
        """

        db = hf.File(self.name, 'r+')  # open the database_path object
        groups = list(db.keys())

        for item in groups:
            if item in mapping:
                db.move(item, mapping[item])

        db.close()

    def load_data(self, path_list: list = None, select_slice: np.s_ = None, dictionary: bool = False):
        """
        Load tensor_values from the database_path for some operation.

        Should be called by the tensor_values fetch class as this will ensure correct loading and pre-loading.
        Returns
        -------

        """
        data: Union[list, dict] = {}
        database = self.open('r')
        if not dictionary:
            data = []
            for item in path_list:
                data.append(tf.convert_to_tensor(database[item][select_slice], dtype=tf.float64))
        if dictionary:
            data = {}
            for item in path_list:
                data[item] = tf.convert_to_tensor(database[item][select_slice], dtype=tf.float64)
        database.close()

        if len(data) == 1:
            if dictionary:
                return data
            else:
                return data[0]
        if dictionary:
            return data
        else:
            return data

    def get_load_time(self, database_path: str = None):
        """
        Calculate the open/close time of the database_path.

        Parameters
        ----------
        database_path : str
                Database path on which to test the time.
        Returns
        -------
        opening time : float
                Time taken to open and close the database_path
        """
        if database_path is None:
            start = time.time()
            database_path = hf.File(self.name, 'r')
            database_path.close()
            stop = time.time()
        else:
            start = time.time()
            database_path = hf.File(database_path, 'r')
            database_path.close()
            stop = time.time()

        return stop - start

    def get_data_size(self, data_path: str, database_path: str = None, system: bool = False) -> tuple:
        """
        Return the size of a dataset as a tuple (n_rows, n_columns, n_bytes)

        Parameters
        ----------
        data_path : str
                path to the tensor_values in the hdf5 database_path.
        database_path: (optional) str
                path to a specific database_path, if None, the class instance database_path will be used
        system : bool
                If true, the row number is the relavent property
        Returns
        -------
        dataset_properties : tuple
                Tuple of tensor_values about the dataset, e.g. (n_rows, n_columns, n_bytes)
        """
        if database_path is None:
            database_path = self.name

        if system:
            with hf.File(database_path, 'r') as db:
                data_tuple = (db[data_path].shape[0], db[data_path].shape[0], db[data_path].nbytes)
        else:
            with hf.File(database_path, 'r') as db:
                data_tuple = (db[data_path].shape[0], db[data_path].shape[1], db[data_path].nbytes)

        return data_tuple
