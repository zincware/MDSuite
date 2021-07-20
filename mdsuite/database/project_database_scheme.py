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


class Experiment(Base):
    """
    Class for the experiment table associated with the Project table.
    """
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    experiment_data = relationship("ExperimentData",
                                   cascade='all delete',
                                   back_populates='experiment')
    computation = relationship("Computations")

    def __repr__(self):
        """
        Representation of the experiment table.

        Returns
        -------
        information : str
                Experiment number and name as an fstring
        """
        return f"{self.id}: {self.name}"


class ExperimentData(Base):
    """
    Class for the experiment data table.

    This table is arbitrarily defined and therefore anything can be added to it.

    Attributes
    ----------
    id : int
            Unique identifier of the row.
    name : str
            name of the property being recorded.
    value : float
            numeric value of the property.
    str_value : str
            String value of the property.
    """
    __tablename__ = 'experiment_data'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    experiment_id = Column(Integer, ForeignKey('experiment.id', ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates='experiment_data')


class Computation(Base):
    """
    Class for the computation table.
    """
    __tablename__ = 'computations'

    id = Column(Integer, primary_key=True)

    experiment_id = Column(Integer, ForeignKey('experiment.id', ondelete="CASCADE"))
    experiment = relationship("Experiment")

    computation_attributes = relationship('ComputationAttributes',
                                         cascade='all delete',
                                         back_populates='computation')
    computation_data = relationship('ComputationData',
                                    cascade='all delete',
                                    back_populates='computation')


class ComputationAttribute(Base):
    """
    Class for the meta data of a computation.
    """
    __tablename__ = 'computation_attributes'

    # Table data
    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    # Relation data
    computation_id = Column(Integer, ForeignKey('computation.id', ondelete="CASCADE"))
    computation = relationship("Computation", back_populates='computation_attributes')


class ComputationData(Base):
    """
    raw computation data of a calculation.
    """
    __tablename__ = 'computation_data'

    id = Column(Integer, primary_key=True)

    value = Column(Float)
    dimension = Column(String)

    # Relation data
    computation_id = Column(Integer, ForeignKey('computation.id', ondelete="CASCADE"))
    computation = relationship("Computation", back_populates='computation_attributes')
