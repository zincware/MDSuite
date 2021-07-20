"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the project database.
"""
import logging
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from .database_scheme import Base, SystemProperty, Data, Subject

log = logging.getLogger(__file__)


class ProjectDatabase:
    """
    Class for the management of the project database.
    """

    def __init__(self, name: str):
        """
        Constructor for the Project database class.

        Parameters
        ----------
        name : str
                Path to the database location.
        """
        self.name = name

        # Database parameters
        self.engine = sa.create_engine(f"sqlite+pysqlite:///{self.name}",
                                       echo=False,
                                       future=True)

        self.Session: sessionmaker
        self.Base = Base

        self.get_session()
        self.build_database()

    def get_session(self):
        """
        Create a session.
        Returns
        -------

        """
        log.debug('Creating the sessionmaker')
        self.Session = sessionmaker(bind=self.engine, future=True)

    def build_database(self):
        """
        Create the database scheme.

        Returns
        -------

        """
        log.debug("Creating the database if it does not exist.")
        self.Base.metadata.create_all(self.engine)