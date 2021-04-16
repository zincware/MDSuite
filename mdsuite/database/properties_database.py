"""
Python module for the properties database.
"""

import sqlalchemy as sql
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import column
from sqlalchemy import table
from sqlalchemy import delete


class PropertiesDatabase:
    """
    A class to control the properties database.
    """

    def __init__(self, name: str):
        """
        Constructor for the PropertiesDatabase class.

        Parameters
        ----------
        name : str
                Name of the database. Should be the full path to the name.
        """
        self.name = name
        self.engine = sql.create_engine(f"sqlite+pysqlite:///{self.name}", echo=False, future=True)

    def build_database(self):
        """
        Build a new database

        Returns
        -------
        Constructs a new database
        """
        with self.engine.begin() as conn:
            stmt = sql.text("CREATE TABLE system_properties (Analysis varchar(255), "
                            "Subject varchar(255), data_range INT, data REAL , uncertainty REAL)")
            conn.execute(stmt)

    def _check_row_existence(self, parameters: dict):
        """
        Check if a row exists.

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500, "data": 1.8e-9}
        Returns
        -------
        result : bool
                True or False depending on existence.
        """
        truth_table = []
        with self.engine.begin() as conn:
            data = table('system_properties', column('Analysis'),
                         column('Subject'), column('data_range'),
                         column('data'), column('uncertainty'))
            stmt = select(data).where(column('Subject') == parameters['Subject'],
                                      column('data_range') == parameters['data_range'])
            for row in conn.execute(stmt):
                truth_table.append(True)

        return any(truth_table)

    def _delete_duplicate_rows(self, parameters: dict):
        """
        Delete duplicate rows

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500, "data": 1.8e-9}
        Returns
        -------
        result : bool
                True or False depending on existence.
        """

        with self.engine.begin() as conn:
            data = table('system_properties', column('Analysis'),
                         column('Subject'), column('data_range'),
                         column('data'), column('uncertainty'))
            stmt = delete(data).where(column('Subject') == parameters['Subject'],
                                      column('data_range') == parameters['data_range'])
            conn.execute(stmt)

    def add_data(self, parameters: dict, delete_duplicate: bool=True):
        """
        Add data to the database

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500, "data": 1.8e-9}
        delete_duplicate : bool
                If true, duplicate enties will be deleted.
        Returns
        -------
        Updates the sql database
        """
        self._delete_duplicate_rows(parameters)
        with self.engine.begin() as conn:
            conn.execute(table('system_properties', column('Analysis'),
                               column('Subject'), column('data_range'),
                               column('data'), column('uncertainty')).insert().values(parameters))
