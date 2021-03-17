"""
Python module for the data fetch class
"""

import numpy as np
import tensorflow as tf
from typing import Callable

from mdsuite.database.database import Database


class DataManager:
    """
    Class for the MDS data fetcher

    Due to the amount of data that needs to be collected and the possibility to optimize repeated loading, a separate
    data fetching class is required. This class manages how data is loaded from the MDS database and optimizes
    processes such as pre-loading and parallel reading.
    """

    def __init__(self, database: Database = None, data_path: list = None, data_range: int = None,
                 batch_number: int = None, batch_size: int = None, ensemble_loop: int = None,
                 correlation_time: int = 1):
        """
        Constructor for the DataManager class

        Parameters
        ----------
        database : Database
                Database object from which data should be loaded
        """
        self.database = database
        self.data_path = data_path

        self.data_range = data_range
        self.batch_number = batch_number
        self.batch_size = batch_size
        self.ensemble_loop = ensemble_loop
        self.correlation_time = correlation_time

    def batch_generator(self) -> tuple:
        """
        Build a generator object for the batch loop
        Returns
        -------
        Returns a generator function and its arguments
        """

        args = (self.batch_number,
                self.batch_size,
                self.database.name,
                self.data_path)

        def generator(batch_number: int, batch_size: int, database: str, data_path: list):
            """
            Generator function for the batch loop.

            Parameters
            ----------
            batch_number : int
                    Number of batches to be looped over
            batch_size : int
                    size of each batch to load
            database : Database
                    database from which to load the data
            data_path : str
                    Path to the data in the database
            Returns
            -------

            """
            database = Database(name=database)
            for batch in range(batch_number):
                start = int(batch*batch_size)
                stop = int(start + batch_size)
                yield database.load_data(data_path, select_slice=np.s_[:, start:stop])

        return generator, args

    def ensemble_generator(self) -> tuple:
        """
        Build a generator for the ensemble loop

        Returns
        -------
        Ensemble loop generator
        """

        args = {tf.cast(self.ensemble_loop, dtype=tf.int64),
                tf.cast(self.correlation_time, dtype=tf.int64),
                tf.cast(self.data_range, dtype=tf.int64)}

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

        return generator, args
