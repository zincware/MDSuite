"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
from __future__ import annotations

import logging

import mdsuite.database.scheme as db
from mdsuite.database.scheme import Project, Experiment, ExperimentAttribute
from mdsuite.utils.database import get_or_create
import pandas as pd
import numpy as np
from dataclasses import asdict
from mdsuite.utils.units import Units

from pathlib import Path
from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from mdsuite import Project

log = logging.getLogger(__name__)


class ExperimentDatabase:
    def __init__(self, project: Project, experiment_name):
        self.project = project
        self.name = experiment_name

        # Property cache
        self._species = None
        self._molecules = None

    def export_property_data(self, parameters: dict) -> List[db.Computation]:
        """
        Export property data from the SQL database.

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na",
                "data_range": 500}
        Returns
        -------
        output : list
                A list of rows represented as dictionaries.
        """
        raise DeprecationWarning(
            "This function has been removed and replaced by queue_database"
        )

    def _set_db(self, name: str, value):
        """Store values in the database

        Parameters
        ----------
        name: str
            Name of the database entry
        value:
            Any serializeable data type that can be written to the database
        """
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            if not isinstance(value, dict):
                value = {"serialized_value": value}
            attribute: ExperimentAttribute = get_or_create(
                ses, ExperimentAttribute, experiment=experiment, name=name
            )
            attribute.data = value
            ses.commit()

    def _get_db(self, name: str, default=None):
        """Load values from the database

        Parameters
        ----------
        name: str
            Name of the datbase entry to query from
        default: default=None
            Default value to yield if not entry is presend

        Returns
        -------
        Any:
            returns the entry that was put in the database, can be any
            json serializeable data

        Notes
        -----
        Internally the values will be converted to dict, so e.g. tuples or sets
         might be converted to lists
        """
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            attribute: ExperimentAttribute = (
                ses.query(ExperimentAttribute)
                .filter(ExperimentAttribute.experiment == experiment)
                .filter(ExperimentAttribute.name == name)
                .first()
            )
            try:
                data = attribute.data
            except AttributeError:
                log.debug(f"Got no database entries for {name}")
                return default
            try:
                return data["serialized_value"]
            except KeyError:
                log.debug(f"Returning a dictionary for {name}")
                return data

    @property
    def active(self):
        """Get the state (activated or not) of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
        return experiment.active

    @active.setter
    def active(self, value):
        """Set the state (activated or not) of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            experiment.active = value
            ses.commit()

    @property
    def species(self):
        """Get species

        Returns
        -------

        dict:
            A dictionary of species such as {Li: {indices: [1, 2, 3], mass: [12.0],
            charge: [0]}}
        """
        if self._species is None:
            with self.project.session as ses:
                experiment = (
                    ses.query(Experiment).filter(Experiment.name == self.name).first()
                )
                self._species = experiment.get_species()

        return self._species

    @species.setter
    def species(self, value: dict):
        """

        Parameters
        ----------
        value

        Notes
        -----

        species = {C: {indices: [1, 2, 3], mass: [12.0], charge: [0]}}

        """
        if value is None:
            return
        self._species = None
        with self.project.session as ses:
            experiment = (
                ses.query(Experiment).filter(Experiment.name == self.name).first()
            )
            for species_name, species_data in value.items():
                species = get_or_create(
                    ses, db.ExperimentSpecies, name=species_name, experiment=experiment
                )
                species.data = species_data
            ses.commit()

    @property
    def molecules(self):
        """Get the molecules dict"""
        if self._molecules is None:
            with self.project.session as ses:
                experiment = (
                    ses.query(Experiment).filter(Experiment.name == self.name).first()
                )
                self._molecules = experiment.get_molecules()
        return self._molecules

    @molecules.setter
    def molecules(self, value):
        """Save the molecules dict to the database"""
        if value is None:
            return
        self._molecules = None
        with self.project.session as ses:
            experiment = (
                ses.query(Experiment).filter(Experiment.name == self.name).first()
            )
            for molecule_name, molecule_data in value.items():
                molecule = get_or_create(
                    ses,
                    db.ExperimentSpecies,
                    name=molecule_name,
                    experiment=experiment,
                    molecule=True,
                )
                molecule.data = molecule_data
            ses.commit()

    # Almost Lazy Properties
    @property
    def box_array(self):
        """Get the sample_rate of the experiment"""
        return self._get_db(name="box_array")

    @box_array.setter
    def box_array(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        if isinstance(value, np.ndarray):
            value = value.tolist()

        self._set_db(name="box_array", value=value)

    # Lazy Properties:
    @property
    def temperature(self):
        """Get the temperature of the experiment"""
        return self._get_db(name="temperature")

    @temperature.setter
    def temperature(self, value):
        """Set the temperature of the experiment"""
        if value is None:
            return
        self._set_db(name="temperature", value=value)

    @property
    def time_step(self):
        """Get the time_step of the experiment"""
        return self._get_db(name="time_step")

    @time_step.setter
    def time_step(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        self._set_db(name="time_step", value=value)

    @property
    def units(self) -> Dict[str, float]:
        """Get the units of the experiment"""
        return self._get_db(name="units")

    @units.setter
    def units(self, value: Units):
        """Set the units of the experiment"""
        if value is None:
            return
        self._set_db(name="units", value=asdict(value))

    @property
    def number_of_configurations(self) -> int:
        """Get the time_step of the experiment"""
        return self._get_db(name="number_of_configurations")

    @number_of_configurations.setter
    def number_of_configurations(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        self._set_db(name="number_of_configurations", value=value)

    @property
    def number_of_atoms(self) -> int:
        """Get the time_step of the experiment"""
        return self._get_db(name="number_of_atoms")

    @number_of_atoms.setter
    def number_of_atoms(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        self._set_db(name="number_of_atoms", value=value)

    @property
    def sample_rate(self):
        """Get the sample_rate of the experiment"""
        return self._get_db(name="sample_rate")

    @sample_rate.setter
    def sample_rate(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        self._set_db(name="sample_rate", value=value)

    @property
    def volume(self):
        """Get the volume of the experiment"""
        return self._get_db(name="volume")

    @volume.setter
    def volume(self, value):
        """Set the volume of the experiment"""
        if value is None:
            return
        self._set_db(name="volume", value=value)

    @property
    def property_groups(self):
        """Get the property groups from the database

        Returns
        -------
        property_groups: dict
                Example: {'Positions': [3, 4, 5], 'Velocities': [6, 7, 8], ...}
        """
        return self._get_db(name="property_groups")

    @property_groups.setter
    def property_groups(self, value):
        """Write the property groups to the database"""
        if value is None:
            return
        self._set_db(name="property_groups", value=value)

    @property
    def read_files(self):
        """

        Returns
        -------
        read_files: list[Path]
            A List of all files that were added to the database already

        """
        return self._get_db(name="read_files", default=[])

    @read_files.setter
    def read_files(self, value):
        """Add a file that has been read to the database

        Does nothing if the file already  exists within the database

        Parameters
        ----------
        value: str, Path
            A filepath that will be added to the database

        """
        if value is None:
            return
        self._set_db(name="read_files", value=value)

    @property
    def radial_distribution_function_state(self) -> bool:
        """Get the radial_distribution_function_state of the experiment

        Returns
        -------
        bool:
            the state of the RDF
        # TODO this method could potentially be replaced by a quick query of the
           database
        """
        return self._get_db(name="radial_distribution_function_state", default=False)

    @radial_distribution_function_state.setter
    def radial_distribution_function_state(self, value):
        """Set the radial_distribution_function_state of the experiment"""
        if value is None:
            return
        self._set_db(name="radial_distribution_function_state", value=value)

    @property
    def simulation_data(self) -> dict:
        """
        Load simulation data from internals.
        If not available try to read them from file

        Returns
        -------
        dict: A dictionary containing all simulation_data

        """
        return self._get_db(name="simulation_data", default={})

    @simulation_data.setter
    def simulation_data(self, value: dict):
        """Update simulation data

        Try to load the data from self.simulation_data_file, update the internals and
        update the yaml file.

        Parameters
        ----------
        value: dict
            A dictionary containing the (new) simulation data

        Returns
        -------
        Updates the internal simulation_data

        """
        if value is None:
            return
        self._set_db(name="simulation_data", value=value)

    @property
    def version(self) -> int:
        """Get the version of the experiment

        Versioning starts at 0 and can be increased by +1 for every added file
        """
        return self._get_db(name="version", default=0)

    @version.setter
    def version(self, value: int):
        """Update the version of the experiment

        Can be used to differentiate between different experiment versions in
        calculations
        """
        if value is None:
            return
        self._set_db(name="version", value=value)
