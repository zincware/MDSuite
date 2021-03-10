"""
Python module for the data fetch class
"""

from mdsuite.database.database import Database


class DataFetcher:
    """
    Class for the MDS data fetcher

    Due to the amount of data that needs to be collected and the possibility to optimize repeated loading, a separate
    data fetching class is required. This class manages how data is loaded from the MDS database and optimizes
    processes such as pre-loading and parallel reading.
    """

    def __init__(self, database: Database):
        """
        Constructor for the DataFetcher class

        Parameters
        ----------
        database : Database
                Database object from which data should be loaded
        """
        self.database = database

