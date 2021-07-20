"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the project database schema.
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import DetachedInstanceError

import numpy as np

Base = declarative_base()


class Project(Base):
    """
    Class for the project table.

    Parameters
    ----------
    experiment : int
            Number of the experiment for which to load data.
    property : str
            Name of the property, e.g. diffusion coefficient.
    analysis : str
            Name of the analysis, e.g. einstein diffusion coefficients.
    data_range : int
            Data range used in the analysis.
    information : str
            Any additional information about the analysis required.
    """
    __table_name__ = 'project'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    experiment = Column(Integer)
    property = Column(String)
    analysis = Column(String)
    data_range = Column(Integer)
    information = Column(String, nullable=True)

    def __init__(self,
                 experiment: int,
                 property: str,
                 analysis: str,
                 data_range: int,
                 information: str):
        """
        Constructor for the Project class.

        Parameters
        ----------
        experiment : int
                    Number of the experiment for which to load data.
        property : str
                Name of the property, e.g. diffusion coefficient.
        analysis : str
                Name of the analysis, e.g. einstein diffusion coefficients.
        data_range : int
                Data range used in the analysis.
        information : str
                Any additional information about the analysis required.
        """
        self.experiment = experiment
        self.property = property
        self.analysis = analysis
        self.data_range = data_range
        self.information = information




