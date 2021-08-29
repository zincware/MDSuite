"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Collection of all used SQLAlchemy Database schemes
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import DetachedInstanceError

import numpy as np

Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=True)

    experiments = relationship("Experiment")


class Experiment(Base):
    """
    Class for the experiment table associated with the Project table.
    """
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String)

    active = Column(Boolean, default=False)
    # Whether this experiment is currently loaded in the project class

    project_id = Column(Integer, ForeignKey('projects.id', ondelete="CASCADE"))
    project = relationship("Project")

    experiment_data = relationship("ExperimentData",
                                   cascade='all, delete',
                                   back_populates='experiment')

    computations = relationship("Computation")
    species = relationship("Species")

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

    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates='experiment_data')

    def __repr__(self):
        if self.value is not None:
            return f"{self.value}"
        elif self.str_value is not None:
            return self.str_value
        else:
            return None


class Species(Base):
    __tablename__ = 'species'
    id = Column(Integer, primary_key=True)
    name = Column(String)

    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates='species')

    species_data = relationship("SpeciesData")

    @property
    def indices(self):
        indices = []
        for species_data in self.species_data:
            if species_data.name == "indices":
                indices.append(int(species_data.value))
        return indices

    @property
    def mass(self):
        mass = []
        for species_data in self.species_data:
            if species_data.name == "mass":
                mass.append(species_data.value)
        return mass

    @property
    def charge(self):
        charge = []
        for species_data in self.species_data:
            if species_data.name == "charge":
                charge.append(species_data.value)
        return charge

#      TODO consider adding species data as a function here!


class SpeciesData(Base):
    __tablename__ = 'species_data'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    species_id = Column(Integer, ForeignKey('species.id', ondelete="CASCADE"))
    species = relationship("Species", back_populates='species_data')


class Computation(Base):
    """
    Class for the computation table.
    """
    __tablename__ = 'computations'

    id = Column(Integer, primary_key=True)

    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete="CASCADE"))
    experiment = relationship("Experiment")

    computation_attributes = relationship('ComputationAttribute',
                                          cascade='all, delete',
                                          back_populates='computation')
    computation_data = relationship('ComputationData',
                                    cascade='all, delete',
                                    back_populates='computation')

    def __repr__(self):
        """
        Representation of the experiment table.

        Returns
        -------
        information : str
                Experiment number and name as an fstring
        """
        return f"{self.id}: {self.experiment_id}"


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
    computation_id = Column(Integer, ForeignKey('computations.id', ondelete="CASCADE"))
    computation = relationship("Computation", back_populates='computation_attributes')

    def __repr__(self):
        return f"{self.name}: {self.value} - {self.str_value}"


class ComputationData(Base):
    """
    raw computation data of a calculation.
    """
    __tablename__ = 'computation_data'

    id = Column(Integer, primary_key=True)

    value = Column(Float)
    uncertainty = Column(Float, nullable=True)
    dimension = Column(String)

    # Relation data
    computation_id = Column(Integer, ForeignKey('computations.id', ondelete="CASCADE"))
    computation = relationship("Computation", back_populates='computation_data')

    def __repr__(self):
        return f"{self.id}: {self.value} ({self.uncertainty}) - {self.dimension}"
