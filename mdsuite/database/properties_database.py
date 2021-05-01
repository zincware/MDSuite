"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Python module for the properties database.
"""

import sqlalchemy as sql
from sqlalchemy import select
from sqlalchemy import column
from sqlalchemy import table
from sqlalchemy import delete
from sqlalchemy import and_


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
        self.table = table('system_properties', column('Property'), column('Analysis'),
                           column('Subject'), column('data_range'),
                           column('data'), column('uncertainty'))

    def build_database(self):
        """
        Build a new database

        Returns
        -------
        Constructs a new database
        """
        with self.engine.begin() as conn:
            stmt = sql.text("CREATE TABLE system_properties (Property varchar(255), Analysis varchar(255), "
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
            stmt = select(self.table).where(column('Subject') == parameters['Subject'],
                                            column('data_range') == parameters['data_range'])
            for _ in conn.execute(stmt):
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
            cond= and_(*[column('Subject') == parameters['Subject'], column('data_range') == parameters['data_range'],
                         column('Analysis') == parameters['Analysis']])
            stmt = delete(self.table).where(cond)
            conn.execute(stmt)

    def add_data(self, parameters: dict, delete_duplicate: bool = True):
        """
        Add data to the database

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500, "data": 1.8e-9}
        delete_duplicate : bool
                If true, duplicate entries will be deleted.
        Returns
        -------
        Updates the sql database
        """
        if delete_duplicate:
            self._delete_duplicate_rows(parameters)
        else:
            if self._check_row_existence(parameters):
                print("Note, an entry with these parameters already exists in the database.")
        with self.engine.begin() as conn:
            conn.execute(self.table.insert().values(parameters))

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
