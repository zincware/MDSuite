"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Python module for the tensor_values fetch class
"""

import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from mdsuite.database.simulation_database import Database


class DataManager:
    """
    Class for the MDS tensor_values fetcher

    Due to the amount of tensor_values that needs to be collected and the possibility to optimize repeated loading, a separate
    tensor_values fetching class is required. This class manages how tensor_values is loaded from the MDS database_path and optimizes
    processes such as pre-loading and parallel reading.
    """

    def __init__(self, database: Database = None, data_path: list = None, data_range: int = None,
                 n_batches: int = None, batch_size: int = None, ensemble_loop: int = None,
                 correlation_time: int = 1, remainder: int = None, atom_selection=np.s_[:],
                 minibatch: bool = False, atom_batch_size : int = None, n_atom_batches: int = None,
                 atom_remainder: int = None, offset: int = 0):
        """
        Constructor for the DataManager class

        Parameters
        ----------
        database : Database
                Database object from which tensor_values should be loaded
        """
        self.database = database
        self.data_path = data_path
        self.minibatch = minibatch
        self.atom_batch_size = atom_batch_size
        self.n_atom_batches = n_atom_batches
        self.atom_remainder = atom_remainder
        self.offset = offset

        self.data_range = data_range
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.remainder = remainder
        self.ensemble_loop = ensemble_loop
        self.correlation_time = correlation_time
        self.atom_selection = atom_selection

    def batch_generator(self, dictionary: bool = False, system: bool = False, remainder: bool = False) -> tuple:
        """
        Build a generator object for the batch loop
        Returns
        -------
        Returns a generator function and its arguments
        """

        args = (self.n_batches,
                self.batch_size,
                self.database.name,
                self.data_path,
                dictionary)

        def generator(batch_number: int, batch_size: int, database: str, data_path: list, dictionary: bool):
            """
            Generator function for the batch loop.

            Parameters
            ----------
            batch_number : int
                    Number of batches to be looped over
            batch_size : int
                    size of each batch to load
            database : Database
                    database_path from which to load the tensor_values
            data_path : str
                    Path to the tensor_values in the database_path
            dictionary : bool
                    If true, tensor_values is returned in a dictionary
            Returns
            -------
            """
            database = Database(name=database)

            _remainder = [1 if remainder else 0][0]
            if self.remainder == 0:
                _remainder = 0
            for batch in range(batch_number + _remainder):
                start = int(batch*batch_size) + self.offset
                stop = int(start + batch_size)
                data_size = tf.cast(batch_size, dtype=tf.int32)
                if batch == batch_number:
                    stop = int(start + self.remainder)
                    data_size = tf.cast(self.remainder, dtype=tf.int16)
                if type(self.atom_selection) is dict:
                    select_slice = {}
                    for item in self.atom_selection:
                        select_slice[item] = np.s_[self.atom_selection[item], start:stop]
                else:
                    select_slice = np.s_[self.atom_selection, start:stop]
                yield database.load_data(data_path,
                                         select_slice=select_slice,
                                         dictionary=dictionary,
                                         d_size=data_size)

        def system_generator(batch_number: int, batch_size: int, database: str, data_path: list, dictionary: bool):
            """
            Generator function for the batch loop.

            Parameters
            ----------
            batch_number : int
                    Number of batches to be looped over
            batch_size : int
                    size of each batch to load
            database : Database
                    database_path from which to load the tensor_values
            data_path : str
                    Path to the tensor_values in the database_path
            dictionary : bool
                    If true, tensor_values is returned in a dictionary
            Returns
            -------

            """
            database = Database(name=database)

            _remainder = [1 if remainder else 0][0]
            if self.remainder == 0:
                _remainder = 0

            for batch in range(batch_number + _remainder):  # +1 for the remainder
                start = int(batch*batch_size) + self.offset
                stop = int(start + batch_size)
                if batch == batch_number:
                    stop = int(start + self.remainder)

                yield database.load_data(data_path, select_slice=np.s_[start:stop], dictionary=dictionary)

        def atom_generator(batch_number: int, batch_size: int, database: str, data_path: list, dictionary: bool):
            """
            Generator function for the batch loop.

            Parameters
            ----------
            batch_number : int
                    Number of batches to be looped over
            batch_size : int
                    size of each batch to load
            database : Database
                    database_path from which to load the tensor_values
            data_path : str
                    Path to the tensor_values in the database_path
            dictionary : bool
                    If true, tensor_values is returned in a dictionary
            Returns
            -------
            """
            database = Database(name=database)
            _atom_remainder = [1 if self.atom_remainder else 0][0]
            for atom_batch in tqdm(range(self.n_atom_batches + _atom_remainder),
                                   total=self.n_atom_batches + _atom_remainder):
                atom_start = atom_batch*self.atom_batch_size
                atom_stop = atom_start + self.atom_batch_size
                if atom_batch == self.n_atom_batches:
                    atom_stop = start + self.atom_remainder
                _remainder = [1 if remainder else 0][0]
                if self.remainder == 0:
                    _remainder = 0
                for batch in range(batch_number + _remainder):
                    start = int(batch*batch_size) + self.offset
                    stop = int(start + batch_size)
                    data_size = tf.cast(batch_size, dtype=tf.int32)
                    if batch == batch_number:
                        stop = int(start + self.remainder)
                        data_size = tf.cast(self.remainder, dtype=tf.int16)
                    if type(self.atom_selection) is dict:
                        print("Atom selection is not available for mini-batched calculations")
                        sys.exit(1)
                    else:
                        select_slice = np.s_[atom_start:atom_stop, start:stop]
                    yield database.load_data(data_path,
                                             select_slice=select_slice,
                                             dictionary=dictionary,
                                             d_size=data_size)

        if system:
            return system_generator, args
        elif self.minibatch:
            return atom_generator, args
        else:
            return generator, args

    def ensemble_generator(self, system: bool = False, dictionary: bool = False) -> tuple:
        """
        Build a generator for the ensemble loop

        Parameters
        ----------
        system : bool
                If true, the system generator is returned.

        Returns
        -------
        Ensemble loop generator
        """

        args = (self.ensemble_loop,
                self.correlation_time,
                self.data_range)

        def generator(ensemble_loop, correlation_time, data_range, data):
            """
            Generator for the ensemble loop
            Parameters
            ----------
            ensemble_loop : int
                    Number of ensembles to loop over
            correlation_time : int
                    Distance between ensembles
            data_range : int
                    Size of each ensemble
            data : tf.data.Dataset
                    Data from which to draw ensembles

            Returns
            -------
            None
            """
            for ensemble in range(ensemble_loop):
                start = ensemble*correlation_time
                stop = start + data_range
                yield data[:, start:stop]

        def system_generator(ensemble_loop, correlation_time, data_range, data):
            """
            Generator for the ensemble loop
            Parameters
            ----------
            ensemble_loop : int
                    Number of ensembles to loop over
            correlation_time : int
                    Distance between ensembles
            data_range : int
                    Size of each ensemble
            data : tf.data.Dataset
                    Data from which to draw ensembles

            Returns
            -------
            None
            """
            for ensemble in range(ensemble_loop):
                start = ensemble*correlation_time
                stop = start + data_range
                yield data[start:stop]

        def dictionary_generator(ensemble_loop, correlation_time, data_range, data_dict):
            """
            Generator for the ensemble loop
            Parameters
            ----------
            ensemble_loop : int
                    Number of ensembles to loop over
            correlation_time : int
                    Distance between ensembles
            data_range : int
                    Size of each ensemble
            data_dict : Dictionary
                    Data from which to draw ensembles

            Returns
            -------
            None
            """
            for ensemble in range(ensemble_loop):
                start = ensemble * correlation_time
                stop = start + data_range
                output_dict = []
                for item in data_dict[:-1]:
                    output_dict[item] = data_dict[item][:, start:stop]
                yield output_dict

        if system:
            return system_generator, args
        elif dictionary:
            return dictionary_generator, args
        else:
            return generator, args
