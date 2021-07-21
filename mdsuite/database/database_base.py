"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Base class for accessing the database
"""
import logging

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.orm.session import Session
from sqlalchemy.engine import Engine

from .project_database_scheme import Base

log = logging.getLogger(__file__)


class DatabaseBase:
    """
    Docstring
    """
    def __init__(self, name: str):
        """

        Parameters
        ----------
        name: str
            name of the database
        """
        self.name = name

        self._engine = None
        self._Session = None

    @property
    def engine(self) -> Engine:
        """Create a SQLAlchemy Engine

        Returns
        -------
        a SQLAlchemy Engine connected to the Database

        """
        if self._engine is None:
            self._engine = sa.create_engine(f"sqlite+pysqlite:///{self.name}",
                                            echo=False,
                                            future=True)
        return self._engine

    @property
    def session(self) -> Session:
        """

        Notes
        -------
        Use with context manager like:
        >>> with self.session as ses:
        >>>     ses.add()

        Returns
            Session that can be used inside a context manager

        """
        return sessionmaker(bind=self.engine, future=True)()

    @property
    def base(self) -> declarative_base:
        """

        Returns
        -------
        Get the declarative base from the Database scheme

        """
        return Base

    def build_database(self):
        """Build the database and get create the tables
        """
        log.debug("Creating the database if it does not exist.")
        self.base.metadata.create_all(self.engine)
