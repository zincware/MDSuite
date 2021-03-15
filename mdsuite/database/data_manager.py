"""
Python module for the data fetch class
"""

import numpy as np

from mdsuite.database.database import Database


class DataFetcher:
    """
    Class for the MDS data fetcher

    Due to the amount of data that needs to be collected and the possibility to optimize repeated loading, a separate
    data fetching class is required. This class manages how data is loaded from the MDS database and optimizes
    processes such as pre-loading and parallel reading.
    """

    def __init__(self, database: Database, data_path: str, data_range: int, batch_number: int, batch_size: int,
                 ensemble_loop: int, correlation_time: int):
        """
        Constructor for the DataFetcher class

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

    def batch_generator(self) -> None:
        """
        Build a generator object for the batch loop
        Returns
        -------
        None, it is called as an assignment in an operation.
        """
        for batch in range(self.batch_number):
            start = int(batch*self.batch_size)
            stop = int(start + self.batch_size)
            yield self.database.load_data([self.data_path], select_slice=np.s_[:, start:stop])

    def ensemble_generator(self):
        """
        Build a generator for the ensemble loop

        Returns
        -------
        Ensemble loop generator
        """
        for ensemble in range(self.ensemble_loop):


