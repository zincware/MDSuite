"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import h5py as hf
import numpy as np
from mdsuite.utils.meta_functions import join_path
from mdsuite.utils.exceptions import DatabaseDoesNotExist
import tensorflow as tf
import time
import pandas as pd


var_names = [
    "Temperature",
    "Time",
    "Thermal_Flux",
    "Stress_visc",
    "Positions",
    "Scaled_Positions",
    "Unwrapped_Positions",
    "Scaled_Unwrapped_Positions",
    "Velocities",
    "Forces",
    "Box_Images",
    "Dipole_Orientation_Magnitude",
    "Angular_Velocity_Spherical",
    "Angular_Velocity_Non_Spherical",
    "Torque",
    "Charge",
    "KE",
    "PE",
    "Stress",
]


class Database:
    """
    Database class

    Databases make up a large part of the functionality of MDSuite and are kept
    fairly consistent in structure. Therefore, the database_path structure we
    are using has a separate class with commonly used methods which act as
    wrappers for the hdf5 database_path.

    Attributes
    ----------
    architecture : str
                The type of the database_path implemented, either a simulation
                database_path, or an analysis database_path.

    name : str
            The name of the database_path in question.
    """

    def __init__(self, architecture: str = "simulation", name: str = "database"):
        """
        Constructor for the database_path class.

        Parameters
        ----------
        architecture : str
                The type of the database_path implemented, either a simulation
                database_path, or an analysis database_path.
        name : str
                The name of the database_path in question.
        """

        self.architecture = architecture  # architecture of database_path
        self.name = name  # name of the database_path

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

    @staticmethod
    def _update_indices(
        data: np.array, reference: np.array, batch_size: int, n_atoms: int
    ):
        """
        Update the indices key of the structure dictionary if the tensor_values must be
        sorted.

        Parameters
        ----------
        data : np.ndarray
        reference : np.ndarray
        batch_size : int
        n_atoms : int

        Returns
        -------

        """

        ids = np.reshape(np.array(data[:, 0]).astype(int), (-1, n_atoms))
        ref_ids = np.argsort(ids, axis=1)
        n_batches = ids.shape[0]

        return (
            ref_ids[:, reference - 1] + (np.arange(n_batches) * n_atoms)[None].T
        ).flatten()

    @staticmethod
    def _build_path_input(structure: dict) -> dict:
        """
        Build an input to a hdf5 database_path from a dictionary

        In many cases, whilst a dict can be passed on to a method, it is not ideal for
        use in the hdf5 database_path. This method takes a dictionary and return a new
        dictionary with the relevant file path.


        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes. e.g.
                {'Na': {'Forces': (200, 5000, 3)},
                'Pressure': (5000, 6), 'Temperature': (5000, 1)} In this case, the last
                 value in the tuple corresponds to the number of components that wil be
                 parsed to the database_path. The value in the dict can also be an
                 integer corresponding to a resize operation such as
                 {'Na': {'velocities' 100}}. In any case, the deepest portion of the
                 dict must be a non-dict object and will be returned as the value of the
                 path to it in the new dictionary.


        Returns
        -------
        architecture : dict
                Corrected path in the hdf5 database_path. e.g. {'/Na/Velocities': 100},
                or {'/Na/Forces': (200, 5000, 3)}

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

    def open(self, mode: str = "a") -> hf.File:
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

    def add_data(
        self,
        data: np.array,
        structure: dict,
        start_index: int,
        batch_size: int,
        tensor: bool = False,
        system_tensor: bool = False,
        flux: bool = False,
        sort: bool = False,
        n_atoms: int = None,
    ):
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

        structure : dict
                Structure of the tensor_values to be loaded into the database_path e.g.
                {'Na/Velocities': {'indices': [1, 3, 7, 8, ... ], 'columns' = [3, 4, 5],
                'length': 500}}
        data : np.array
                Data to be loaded in.
        sort : bool
                If true, tensor_values is sorted before being dumped into the
                database_path.
        n_atoms : int
                Necessary if the sort function is called. Total number of atoms in the
                experiment.
        Returns
        -------
        Adds tensor_values to the database_path
        """

        with hf.File(self.name, "r+") as database:
            stop_index = start_index + batch_size  # get the stop index
            for item in structure:
                if tensor:
                    database[item][:, start_index:stop_index, :] = data[:, :, 0:3]
                elif system_tensor:
                    database[item][start_index:stop_index, :] = data[:, 0:3]
                elif flux:
                    database[item][start_index:stop_index, :] = data[
                        structure[item]["indices"]
                    ][
                        np.s_[
                            :,
                            structure[item]["columns"][0] : structure[item]["columns"][
                                -1
                            ]
                            + 1,
                        ]
                    ].astype(
                        float
                    )
                else:
                    database[item][:, start_index:stop_index, :] = self._get_data(
                        data, structure, item, batch_size, sort, n_atoms=n_atoms
                    )

    def _get_data(
        self,
        data: np.array,
        structure: dict,
        item: str,
        batch_size: int,
        sort: bool = False,
        n_atoms: int = None,
    ):
        """
        Fetch tensor_values with some format from a large array.

        Returns
        -------

        """
        if sort:
            indices = self._update_indices(
                data, structure[item]["indices"], batch_size, n_atoms
            )
            return (
                data[indices][
                    np.s_[
                        :,
                        structure[item]["columns"][0] : structure[item]["columns"][-1]
                        + 1,
                    ]
                ]
                .astype(float)
                .reshape(
                    (
                        structure[item]["length"],
                        batch_size,
                        len(structure[item]["columns"]),
                    ),
                    order="F",
                )
            )
        else:
            indices = structure[item]["indices"]
            return (
                data[indices][
                    np.s_[
                        :,
                        structure[item]["columns"][0] : structure[item]["columns"][-1]
                        + 1,
                    ]
                ]
                .astype(float)
                .reshape(
                    (
                        structure[item]["length"],
                        batch_size,
                        len(structure[item]["columns"]),
                    ),
                    order="F",
                )
            )

    def resize_dataset(self, structure: dict):
        """
        Resize a dataset so more tensor_values can be added

        Parameters
        ----------
        structure : dict
                path to the dataset that needs to be resized. e.g.
                {'Na': {'velocities': (32, 100, 3)}}
                will resize all 'x', 'y', and 'z' datasets by 100 entries.

        Returns
        -------

        """
        # ensure the database_path already exists
        try:
            database = hf.File(self.name, "r+")
        except DatabaseDoesNotExist:
            raise DatabaseDoesNotExist

        # construct the architecture dict
        architecture = self._build_path_input(structure=structure)

        # Check for a type error in the dataset information
        for identifier in architecture:
            dataset_information = architecture[identifier]
            try:
                if type(dataset_information) is not tuple:
                    print("Invalid input for dataset generation")
                    raise TypeError
            except TypeError:
                raise TypeError

            # get the correct maximum shape for the dataset -- changes if an
            # experiment property or an atomic property
            if len(dataset_information[:-1]) == 1:
                axis = 0
                expansion = dataset_information[0] + database[identifier].shape[0]
            else:
                axis = 1
                expansion = dataset_information[1] + database[identifier].shape[1]
            database[identifier].resize(expansion, axis)

    def initialize_database(self, structure: dict):
        """
        Build a database_path with a general structure.

        Note, this method WILL overwrite a pre-existing database_path. This is because
        it is only to be called on the initial construction of an experiment class and
        the first addition of tensor_values to it.


        Parameters
        ----------
        structure : dict
                General structure of the dictionary with relevant dataset sizes.
                e.g. {'Na': {'Forces': (200, 5000, 3)}, 'Pressure': (5000, 6),
                'Temperature': (5000, 1)} In this case, the last value in the tuple
                 corresponds to the number of components that wil be parsed to the
                 database_path.
        Returns
        -------

        """

        self.add_dataset(structure)  # add a dataset to the groups

    def add_dataset(self, structure: dict):
        """
        Add a dataset of the necessary size to the database_path

        Just as a separate method exists for building the group structure of the hdf5
        database_path, so too do we include a separate method for adding a dataset.
        This is so datasets can be added not just upon the initial construction of the
        database_path, but also if tensor_values is added in the future that should
        also be stored. This method will assume that a group has already been built,
        although this is not necessary for HDF5, the separation of the actions is good
        practice.

        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database_path.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}

        Returns
        -------
        Updates the database_path directly.
        """

        with hf.File(self.name, "a") as database:
            architecture = self._build_path_input(
                structure
            )  # get the correct file path
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

                if len(dataset_information[:-1]) == 1:
                    vector_length = dataset_information[-1]
                    max_shape = (None, vector_length)
                else:
                    max_shape = list(dataset_information)
                    max_shape[1] = None
                    max_shape = tuple(max_shape)

                database.create_dataset(
                    dataset_path,
                    dataset_information,
                    maxshape=max_shape,
                    scaleoffset=5,
                    chunks=True,
                )

    def _add_group_structure(self, structure: dict):
        """
        Add a simple group structure to a database_path.
        This method will take an input structure and build the required group structure
        in the hdf5 database_path. This will NOT however instantiate the dataset for the
        structure, only the group hierarchy.


        Parameters
        ----------
        structure : dict
                Structure of a single property to be added to the database_path.
                e.g. {'Na': {'Forces': (200, 5000, 3)}}

        Returns
        -------
        Updates the database_path directly.
        """

        with hf.File(self.name, "a") as database:
            # Build file paths for the addition.
            architecture = self._build_path_input(structure=structure)
            for item in list(architecture):
                if item in database:
                    print("Group structure already exists")
                else:
                    database.create_group(item)

    def get_memory_information(self) -> dict:
        """
        Get memory information from the database_path

        Returns
        -------
        memory_database : dict
                A dictionary of the memory information of the groups in the
                database_path
        """
        with hf.File(self.name, "r") as database:
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
        with hf.File(self.name, "r") as database_object:
            keys = []
            database_object.visit(
                lambda item: keys.append(database_object[item].name)
                if type(database_object[item]) is hf.Dataset
                else None
            )
            path = f"/{path}"  # add the / to avoid name overlapping

            response = any(list(item.endswith(path) for item in keys))
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

        # db = hf.File(self.name, 'r+')  # open the database_path object
        with hf.File(self.name, "r+") as db:
            groups = list(db.keys())

            for item in groups:
                if item in mapping:
                    db.move(item, mapping[item])

    def load_data(
        self,
        path_list: list = None,
        select_slice: np.s_ = np.s_[:],
        dictionary: bool = False,
        scaling: list = None,
        d_size: int = None,
    ):
        """
        Load tensor_values from the database_path for some operation.

        Should be called by the tensor_values fetch class as this will ensure
        correct loading and pre-loading.

        Returns
        -------

        """
        if scaling is None:
            scaling = [1 for _ in range(len(path_list))]

        with hf.File(self.name, "r") as database:
            data = {}
            for i, item in enumerate(path_list):
                if type(select_slice) is dict:
                    my_slice = select_slice[item]
                else:
                    my_slice = select_slice

                data[item] = (
                    tf.convert_to_tensor(database[item][my_slice], dtype=tf.float64)
                    * scaling[i]
                )
            data[str.encode("data_size")] = d_size

            # else:
            #     data = []
            #     for i, item in enumerate(path_list):
            #         data.append(
            #             tf.convert_to_tensor(
            #                 database[item][select_slice], dtype=tf.float64
            #             )
            #             * scaling[i]
            #         )

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
            database_path = hf.File(self.name, "r")
            database_path.close()
            stop = time.time()
        else:
            start = time.time()
            database_path = hf.File(database_path, "r")
            database_path.close()
            stop = time.time()

        return stop - start

    def get_data_size(
        self, data_path: str, database_path: str = None, system: bool = False
    ) -> tuple:
        """
        Return the size of a dataset as a tuple (n_rows, n_columns, n_bytes)

        Parameters
        ----------
        data_path : str
                path to the tensor_values in the hdf5 database_path.
        database_path: (optional) str
                path to a specific database_path, if None, the class instance
                database_path will be used
        system : bool
                If true, the row number is the relevant property

        Returns
        -------
        dataset_properties : tuple
                Tuple of tensor_values about the dataset, e.g.
                (n_rows, n_columns, n_bytes)
        """
        if database_path is None:
            database_path = self.name

        if system:
            with hf.File(database_path, "r") as db:
                data_tuple = (
                    db[data_path].shape[0],
                    db[data_path].shape[0],
                    db[data_path].nbytes,
                )
        else:
            with hf.File(database_path, "r") as db:
                data_tuple = (
                    db[data_path].shape[0],
                    db[data_path].shape[1],
                    db[data_path].nbytes,
                )

        return data_tuple

    def get_database_summary(self):
        """
        Print a summary of the database properties.
        Returns
        -------
        summary : list
                A list of properties that are in the database
        """
        dump_list = []
        database = hf.File(self.name, "r")
        initial_list = list(database.keys())
        for item in var_names:
            if item in initial_list:
                dump_list.append(item)
        for item in initial_list:
            sub_items = list(database[item].keys())
            for var in var_names:
                if var in sub_items:
                    dump_list.append(var)
        database.close()
        return np.unique(dump_list)
