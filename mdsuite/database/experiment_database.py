"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the experiment database.
"""
from __future__ import annotations

import logging

from mdsuite.database.scheme import Project, Experiment, ExperimentData, Species, SpeciesData
from mdsuite.utils.database import get_or_create

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Project

log = logging.getLogger(__file__)


class ExperimentDatabase:
    def __init__(self, project: Project, experiment_name):
        self.project = project
        self.experiment_name = experiment_name

    @property
    def active(self):
        """Get the state (activated or not) of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
        return experiment.active

    @active.setter
    def active(self, value):
        """Set the state (activated or not) of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            experiment.active = value
            ses.commit()

    @property
    def temperature(self):
        """Get the temperature of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            temperature = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "temperature").first()
        try:
            return temperature.value
        except AttributeError:
            return None

    @temperature.setter
    def temperature(self, value):
        """Set the temperature of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            temperature: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="temperature")
            temperature.value = value
            ses.commit()

    @property
    def time_step(self):
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            time_step = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "time_step").first()
        try:
            return time_step.value
        except AttributeError:
            return None

    @time_step.setter
    def time_step(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            time_step: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="time_step")
            time_step.value = value
            ses.commit()

    @property
    def unit_system(self):
        """Get the unit_system of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            unit_system = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "unit_system").first()
        try:
            return unit_system.str_value
        except AttributeError:
            return None

    @unit_system.setter
    def unit_system(self, value):
        """Set the unit_system of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            unit_system: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="unit_system")
            unit_system.str_value = value
            ses.commit()

    @property
    def number_of_configurations(self) -> int:
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            number_of_configurations = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "number_of_configurations").first()
        try:
            return int(number_of_configurations.value)
        except AttributeError:
            return None

    @number_of_configurations.setter
    def number_of_configurations(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            number_of_configurations: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment,
                                                                     name="number_of_configurations")
            number_of_configurations.value = value
            ses.commit()

    @property
    def number_of_atoms(self) -> int:
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            number_of_atoms = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "number_of_atoms").first()
        try:
            return int(number_of_atoms.value)
        except AttributeError:
            return None

    @number_of_atoms.setter
    def number_of_atoms(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            number_of_atoms: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment,
                                                            name="number_of_atoms")
            number_of_atoms.value = value
            ses.commit()

    @property
    def sample_rate(self):
        """Get the sample_rate of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            sample_rate = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "sample_rate").first()
        try:
            return sample_rate.value
        except AttributeError:
            return None

    @sample_rate.setter
    def sample_rate(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            sample_rate: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="sample_rate")
            sample_rate.value = value
            ses.commit()

    @property
    def volume(self):
        """Get the volume of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            volume = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "volume").first()
        try:
            return volume.value
        except AttributeError:
            return None

    @volume.setter
    def volume(self, value):
        """Set the volume of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            volume: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="volume")
            volume.value = value
            ses.commit()

    @property
    def box_array(self):
        """Get the sample_rate of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            box_arrays = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name.startswith("box_array")).all()

            box_array = [box_side.value for box_side in box_arrays]

        return box_array

    @box_array.setter
    def box_array(self, value):
        """Set the time_step of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            for idx, box_length in enumerate(value):
                sample_rate: ExperimentData = get_or_create(
                    ses, ExperimentData, experiment=experiment, name=f"box_array_{idx}")
                sample_rate.value = box_length
            ses.commit()

    @property
    def species(self):
        species_dict = {}
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.experiment_name).first()
            for species in experiment.species:
                species_dict.update({
                    species.name: {
                        "indices": species.indices,
                        "mass": species.mass,
                        "charge": species.charge
                    }
                })

        return species_dict

    @species.setter
    def species(self, value):
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
        with self.project.session as ses:
            for species_name in value:
                species = get_or_create(ses, Species, name=species_name)
                species.experiment = ses.query(Experiment).filter(Experiment.name == self.experiment_name).first()
                for species_attr, species_values in value[species_name].items():
                    for idx, species_value in enumerate(species_values):
                        species_data = get_or_create(
                            ses, SpeciesData,
                            name=species_attr, species=species, value=species_value
                        )

            ses.commit()

    @property
    def property_groups(self):
        """Get the property groups from the database

        Returns
        -------
        property_groups: dict
                Example: {'Positions': [3, 4, 5], 'Velocities': [6, 7, 8], ...}
        """
        property_groups = {}
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.experiment_name).first()
            for experiment_data in experiment.experiment_data:
                if experiment_data.name.startswith('property_group_'):
                    property_group = experiment_data.name.split('property_group_', 1)[1]
                    # everything after, max 1 split
                    group_values = property_groups.get(property_group, [])
                    group_values.append(int(experiment_data.value))
                    property_groups[property_group] = group_values

        return property_groups

    @property_groups.setter
    def property_groups(self, value):
        """Write the property groups to the database"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.experiment_name).first()
            for group in value:
                for group_value in value[group]:
                    get_or_create(ses, ExperimentData, name=f"property_group_{group}", value=group_value,
                                  experiment=experiment)

            ses.commit()

    @property
    def read_files(self):
        """

        Returns
        -------
        read_files: list[Path]
            A List of all files that were added to the database already

        """
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.experiment_name).first()
            read_files = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "read_file").all()
            read_files = [Path(file.str_value) for file in read_files]
        return read_files

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
        if isinstance(value, Path):
            value = value.as_posix()
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.experiment_name).first()
            get_or_create(ses, ExperimentData, name="read_file", str_value=value, experiment=experiment)
            ses.commit()
