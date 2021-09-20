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
from dataclasses import dataclass, field, asdict

from typing import List

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

    experiment_attributes = relationship("ExperimentAttribute",
                                         cascade='all, delete',
                                         back_populates='experiment')

    computations = relationship("Computation")
    # species = relationship("Species")

    def __repr__(self):
        """
        Representation of the experiment table.

        Returns
        -------
        information : str
                Experiment number and name as an fstring
        """
        return f"{self.id}: {self.name}"

    @property
    def species(self):
        species_dict = {}
        for experiment_attribute in self.experiment_attributes:
            if experiment_attribute.name == "species":
                species_dict.update(experiment_attribute.species_data)

        return species_dict


class ExperimentAttribute(Base):
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
    __tablename__ = 'experiment_attributes'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates='experiment_attributes')

    experiment_attribute_lists = relationship(
        "ExperimentAttributeList",  # Name of the related class
        cascade='all, delete',
        back_populates='experiment_attribute'  # Attribute of the related class/relationship
    )

    def __repr__(self):
        if self.value is not None:
            return f"{self.value}"
        elif self.str_value is not None:
            return self.str_value
        else:
            return f"{self.name}"

    @property
    def species_data(self) -> dict:
        """If the object is of type species, get the species information


        Returns
        -------

        A dictionary of type {name: {indices: [...], {mass: ...}}
        """

        if self.name != "species":
            raise ValueError(f"Object with name {self.name} does not have species_data!")

        @dataclass
        class SpeciesAttributes:
            """All attributes a species object has

            This is required to distinguish between iterables and non-iterables
            """
            indices: List[int] = field(default_factory=list)
            mass: list = field(default_factory=list)
            charge: list = field(default_factory=list)
            particle_density: float = None
            molar_fraction: float = None

        species_dict = asdict(SpeciesAttributes())
        name = None

        for species_data in self.experiment_attribute_lists:
            if species_data.name in species_dict:
                if isinstance(species_dict[species_data.name], list):
                    if species_data.name == "indices":
                        species_dict[species_data.name].append(int(species_data.value))
                    else:
                        species_dict[species_data.name].append(species_data.value)
                else:
                    species_data[species_data.name] = species_data.value
            elif species_data.name == "name":
                name = species_data.str_value

        return {name: species_dict}


class ExperimentAttributeList(Base):
    """Store lists of data for ExperimentAttributes"""

    __tablename__ = 'experiment_attribute_lists'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    experiment_attribute_id = Column(Integer, ForeignKey('experiment_attributes.id', ondelete="CASCADE"))
    experiment_attribute = relationship("ExperimentAttribute", back_populates='experiment_attribute_lists')


# class Species(Base):
#     __tablename__ = 'species'
#     id = Column(Integer, primary_key=True)
#     name = Column(String)
#
#     experiment_id = Column(Integer, ForeignKey('experiments.id', ondelete="CASCADE"))
#     experiment = relationship("Experiment", back_populates='species')
#
#     species_data = relationship("SpeciesData")
#
#     @property
#     def indices(self) -> list:
#         indices = []
#         for species_data in self.species_data:
#             if species_data.name == "indices":
#                 indices.append(int(species_data.value))
#         return indices
#
#     @property
#     def mass(self) -> list:
#         mass = []
#         for species_data in self.species_data:
#             if species_data.name == "mass":
#                 mass.append(species_data.value)
#         return mass
#
#     @property
#     def charge(self) -> list:
#         charge = []
#         for species_data in self.species_data:
#             if species_data.name == "charge":
#                 charge.append(species_data.value)
#         return charge
#
#     @property
#     def particle_density(self) -> float:
#         for species_data in self.species_data:
#             if species_data.name == "particle_density":
#                 return species_data.value
#         return None
#
#     @property
#     def molar_fraction(self) -> float:
#         for species_data in self.species_data:
#             if species_data.name == "molar_fraction":
#                 return species_data.value
#         return None
#
#
# #      TODO consider adding species data as a function here!
#
#
# class SpeciesData(Base):
#     __tablename__ = 'species_data'
#     id = Column(Integer, primary_key=True)
#     name = Column(String)
#     value = Column(Float, nullable=True)
#     str_value = Column(String, nullable=True)
#
#     species_id = Column(Integer, ForeignKey('species.id', ondelete="CASCADE"))
#     species = relationship("Species", back_populates='species_data')


class Computation(Base):
    """
    Class for the computation table.
    """
    __tablename__ = 'computations'

    id = Column(Integer, primary_key=True)
    name = Column(String, default="Computation")

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
        return f"Exp{self.experiment_id}_{self.name}_{self.id}"

    @property
    def data_dict(self) -> dict:
        """

        Returns
        -------
        data_dict: dict
            A dictionary of the type {x: [1, 2, 3], y: [5, 6, 7], ...}
            where the keys are defined by computation_data.dimension

        """
        data_dict = {}
        for data in self.computation_data:
            data_list = data_dict.get(data.dimension, [])
            data_list.append(data.value)
            data_dict[data.dimension] = data_list

        return data_dict

    @property
    def data_range(self) -> int:
        """Get the data_range stored in computation_attributes"""
        for comp_attr in self.computation_attributes:
            if comp_attr.name == "data_range":
                return int(comp_attr.value)

    @property
    def subjects(self) -> list:
        """Get the subjects stored in computation_attributes"""
        subjects = []
        for comp_attr in self.computation_attributes:
            if comp_attr.name == "subject":
                subjects.append(comp_attr.str_value)
        return subjects


# class ComputationSpecies(Base):
#     """Class for species in the middle between computations and computation data"""
#     __tablename__ = "computation_species"
#     id = Column(Integer, primary_key=True)
#     species = ...


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
