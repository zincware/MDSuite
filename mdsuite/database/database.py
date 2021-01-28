"""
Class for database objects and all of their operations
"""


class Database:
    """
    Database class

    Databases make up a large part of the functionality of MDSuite and are kept fairly consistent in structure.
    Therefore, the database structure we are using has a separate class with commonly used methods which act as
    wrappers for the hdf5 database.

    Attributes
    ----------
    type : str
                The type of the database implemented, either a simulation database, or an analysis database.
    name : str
            The name of the database in question.
    """

    def __init__(self, type='simulation', name='database'):
        """
        Constructor for the database class.

        Parameters
        ----------
        type : str
                The type of the database implemented, either a simulation database, or an analysis database.
        name : str
                The name of the database in question.
        """

        self.type = type  # type of database
        self.name = name  # name of the database

    def add_data(self):
        """
        Add a set of data to the database.

        Returns
        -------

        """
        pass

    def _resize_dataset(self):
        """
        Resize a dataset so more data can be added

        Returns
        -------

        """
        pass

    def _initialize_database(self):
        """
        Build a database with a general structure

        Returns
        -------

        """
        pass

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
