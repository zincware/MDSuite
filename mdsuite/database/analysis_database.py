"""
Python module for the Analysis database class.
"""

import sqlalchemy as sql
from sqlalchemy import select
from sqlalchemy import column
from sqlalchemy import table
from sqlalchemy import delete
from sqlalchemy import and_
import pandas as pd


class AnalysisDatabase:
    """
    Class of the MDSuite Analysis database.
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
        self.engine = sql.create_engine(f"sqlite+pysqlite:///{self.name}", echo=False, future=False)

    def build_database(self):
        """
        Build a new database

        Returns
        -------
        Constructs a new database
        """
        with self.engine.begin() as conn:
            print("Constructed new database")

    def _check_table_existence(self, parameters: dict):
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
            stmt = select(self.table).where(column('Subject') == parameters['Subject'],
                                            column('data_range') == parameters['data_range'])
            for _ in conn.execute(stmt):
                truth_table.append(True)

        return any(truth_table)

    def add_data(self, name: str, data_frame: pd.DataFrame):
        """
        Add data to the database

        Parameters
        ----------

        Returns
        -------
        Updates the sql database
        """
        data_frame.to_sql(name, self.engine, if_exists='replace')

    def load_data(self, parameters: dict):
        """
        Load some data from the database.
        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion",
                 "Subject": "Na",
                 "data_range": 500}

        Returns
        -------
        output : list
                All rows matching the parameters represented as a dictionary.
        """
        output = []
        with self.engine.begin() as conn:
            cond = and_(*[column(item).ilike(parameters[item]) for item in parameters])
            stmt = select(self.table).where(cond)
            for i, row in enumerate(conn.execute(stmt)):
                output.append(dict(row))

        return output
