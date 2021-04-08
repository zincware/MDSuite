"""
Python module for the tensor_values fetch class
"""

import numpy as np
import tensorflow as tf

from mdsuite.database.database import Database


class DataManager:
    """
    Class for the MDS tensor_values fetcher

    Due to the amount of tensor_values that needs to be collected and the possibility to optimize repeated loading, a separate
    tensor_values fetching class is required. This class manages how tensor_values is loaded from the MDS database_path and optimizes
    processes such as pre-loading and parallel reading.
    """

    def __init__(self, database: Database = None, data_path: list = None, data_range: int = None,
                 n_batches: int = None, batch_size: int = None, ensemble_loop: int = None,
                 correlation_time: int = 1, remainder: int = None):
        """
        Constructor for the DataManager class

        Parameters
        ----------
        database : Database
                Database object from which tensor_values should be loaded
        """
        self.database = database
        self.data_path = data_path

        self.data_range = data_range
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.remainder = remainder
        self.ensemble_loop = ensemble_loop
        self.correlation_time = correlation_time

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

            for batch in range(batch_number + _remainder):
                start = int(batch*batch_size)
                stop = int(start + batch_size)
                if batch == batch_number:
                    stop = int(start + self.remainder)

                # print(f'{start}:{stop}:{database.load_data(data_path, select_slice=np.s_[:, start:stop], dictionary=dictionary).shape}')

                yield database.load_data(data_path, select_slice=np.s_[:, start:stop], dictionary=dictionary)

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

            for batch in range(batch_number + _remainder):  # +1 for the remainder
                start = int(batch*batch_size)
                stop = int(start + batch_size)
                if batch == batch_number:
                    stop = int(start + self.remainder)

                # print(f'{start}:{stop}:{database.load_data(data_path, select_slice=np.s_[start:stop], dictionary=dictionary).shape}')

                yield database.load_data(data_path, select_slice=np.s_[start:stop], dictionary=dictionary)

        if system:
            return system_generator, args
        else:
            return generator, args

    def ensemble_generator(self, system: bool = False) -> tuple:
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

        if system:
            return system_generator, args
        else:
            return generator, args
