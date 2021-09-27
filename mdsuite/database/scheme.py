"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Collection of all used SQLAlchemy Database schemes
"""
import logging

from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Boolean, Table
from sqlalchemy.orm import relationship
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
from mdsuite.utils import config
from .types import MutableDict, JSONEncodedDict

from typing import List

log = logging.getLogger(__name__)

Base = declarative_base()


# TODO consider using lazy = True instead of querying data_dict!
# TODO consider serializing some of the computation data

class SpeciesAssociation(Base):
    """Connection between Computations and Experiment Species"""
    __tablename__ = "species_association"
    computation_results_id = Column(ForeignKey('computation_results.id'), primary_key=True)
    experiment_species_id = Column(ForeignKey('experiment_species.id'), primary_key=True)

    count = Column(Integer, default=1) # how often a species occurs, e.g. Na - Na - Cl ADF would be 2, 1

    computation_result = relationship("ComputationResult", back_populates="species")
    species = relationship("ExperimentSpecies", back_populates="computation_result")


class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    description = Column(String, nullable=True)

    experiments = relationship("Experiment")


class Experiment(Base):
    """
    Class for the experiment table associated with the Project table.
    """

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    active = Column(Boolean, default=False)
    # Whether this experiment is currently loaded in the project class

    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"))
    project = relationship("Project")

    experiment_attributes = relationship(
        "ExperimentAttribute", cascade="all, delete", back_populates="experiment"
    )

    computations = relationship("Computation")

    species = relationship("ExperimentSpecies")

    def __repr__(self):
        """
        Representation of the experiment table.

        Returns
        -------
        information : str
                Experiment number and name as an fstring
        """
        return f"{self.id}: {self.name}"

    def get_species(self) -> dict:
        """Get the species information for the experiment"""
        species_dict = {}
        for species in self.species:
            if species.molecule:
                continue
            species: ExperimentSpecies
            species_dict[species.name] = species.data

        return species_dict

    def get_molecules(self) -> dict:
        """Get the molecules information for the experiment"""
        molecule_dict = {}
        for molecule in self.species:
            if molecule.molecule:
                molecule: ExperimentSpecies
                molecule_name = molecule.name
                molecule_dict[molecule_name] = molecule.data

        return molecule_dict


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

    __tablename__ = "experiment_attributes"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates="experiment_attributes")

    experiment_attribute_lists = relationship(
        "ExperimentAttributeList",  # Name of the related class
        cascade="all, delete",
        back_populates="experiment_attribute",  # Attribute of the related class/relationship
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
        log.debug("Accessing species data")
        if self.name != "species":
            raise ValueError(
                f"Object with name {self.name} does not have species_data!"
            )

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

        for species_data in self.experiment_attribute_lists:
            if species_data.name in species_dict:
                if isinstance(species_dict[species_data.name], list):
                    if species_data.name == "indices":
                        species_dict[species_data.name].append(int(species_data.value))
                    else:
                        species_dict[species_data.name].append(species_data.value)
                else:
                    species_data[species_data.name] = species_data.value

        return {self.str_value: species_dict}


class ExperimentAttributeList(Base):
    """Store lists of data for ExperimentAttributes"""

    __tablename__ = 'experiment_attribute_lists'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    experiment_attribute_id = Column(Integer, ForeignKey('experiment_attributes.id', ondelete="CASCADE"))
    experiment_attribute = relationship("ExperimentAttribute", back_populates='experiment_attribute_lists')


class ExperimentSpecies(Base):
    """Table for storing species information

    This table is used to store species and molecule information that can be related to a specific experiment

    """

    __tablename__ = "experiment_species"
    id = Column(Integer, primary_key=True)

    name = Column(String)
    data = Column(MutableDict.as_mutable(JSONEncodedDict))
    molecule = Column(Boolean, default=False)

    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    experiment = relationship("Experiment", back_populates="species")

    computation_result = relationship("SpeciesAssociation", back_populates="species")

    def __repr__(self):
        return f"{self.name}_obj"


class Computation(Base):
    """Class for the computation table."""

    __tablename__ = "computations"

    id = Column(Integer, primary_key=True)
    name = Column(String, default="Computation")

    experiment_id = Column(Integer, ForeignKey("experiments.id", ondelete="CASCADE"))
    experiment = relationship("Experiment")

    computation_attributes = relationship(
        "ComputationAttribute", cascade="all, delete", back_populates="computation"
    )
    computation_data = relationship(
        "ComputationData", cascade="all, delete", back_populates="computation"
    )
    computation_results = relationship(
        "ComputationResult", cascade="all, delete", back_populates="computation", lazy=True
    )

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
        species_dict: dict
            A dictionary of the type
            {
                Li:
                    {
                        a: 1.2,
                        uncert: 0.1,
                        time: [1, 2, 3, ],
                        msd: [0.1, 0.3, 0.7]
                    },
                Cl:
                    {
                        a: 1.2,
                        uncert: 0.1,
                        time: [1, 2, 3, ],
                        msd: [0.1, 0.3, 0.7]
                    },
            }
            where the keys are defined by species (multiple species are joined by "_") and the dimension argument
            of the computation_data

        """
        species_dict = {}
        for result in self.computation_results:
            result: ComputationResult
            species_keys_list = []
            for species_associate in result.species:
                species_associate: SpeciesAssociation
                species_keys_list += species_associate.count * [species_associate.species.name]
            species_keys = "_".join(species_keys_list)
            # iterating over associates
            species_dict[species_keys] = result.data

        return species_dict

    @property
    def data_range(self) -> int:
        """Get the data_range stored in computation_attributes"""
        for comp_attr in self.computation_attributes:
            if comp_attr.name == "data_range":
                return int(comp_attr.str_value)

    @property
    def subjects(self) -> list:
        """Get the subjects stored in computation_attributes"""
        subjects = []
        for comp_attr in self.computation_attributes:
            if comp_attr.name == "subject":
                subjects.append(comp_attr.str_value)
        return subjects


class ComputationAttribute(Base):
    """
    Class for the meta data of a computation.
    """

    __tablename__ = "computation_attributes"

    # Table data
    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float, nullable=True)
    str_value = Column(String, nullable=True)

    # Relation data
    computation_id = Column(Integer, ForeignKey("computations.id", ondelete="CASCADE"))
    computation = relationship("Computation", back_populates="computation_attributes")

    def __repr__(self):
        return f"{self.name}: {self.value} - {self.str_value}"


class ComputationData(Base):
    """
    raw computation data of a calculation.
    """

    __tablename__ = "computation_data"

    id = Column(Integer, primary_key=True)

    value = Column(Float)
    uncertainty = Column(Float, nullable=True)
    dimension = Column(String)

    # Relation data
    computation_id = Column(Integer, ForeignKey("computations.id", ondelete="CASCADE"))
    computation = relationship("Computation", back_populates="computation_data")

    computation_species = relationship(
        "ComputationSpecies",  # Name of the related class
        cascade="all, delete",
        back_populates="computation_data",  # Attribute of the related class/relationship
    )

    def __repr__(self):
        return f"{self.id}: {self.value} ({self.uncertainty}) - {self.dimension}"


class ComputationSpecies(Base):
    """Species associated with computation data"""

    __tablename__ = "computation_species"

    id = Column(Integer, primary_key=True)
    name = Column(String)

    computation_data_id = Column(
        Integer, ForeignKey("computation_data.id", ondelete="CASCADE")
    )
    computation_data = relationship(
        "ComputationData", back_populates="computation_species"
    )


class ComputationResult(Base):
    """
    raw computation data of a calculation.
    """

    __tablename__ = "computation_results"

    id = Column(Integer, primary_key=True)

    data = Column(MutableDict.as_mutable(JSONEncodedDict))

    # Relation data
    computation_id = Column(Integer, ForeignKey("computations.id", ondelete="CASCADE"))
    computation = relationship("Computation", back_populates="computation_results")

    # Many <-> Many
    species = relationship("SpeciesAssociation", back_populates="computation_result")
