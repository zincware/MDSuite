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

import mdsuite.database.scheme as db
from mdsuite.database.scheme import Project, Experiment, ExperimentData, Species, SpeciesData
from mdsuite.utils.database import get_or_create

from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from mdsuite import Project

log = logging.getLogger(__name__)


class ExperimentDatabase:
    def __init__(self, project: Project, experiment_name):
        self.project = project
        self.name = experiment_name

    def export_property_data(self, parameters: dict) -> List[db.Computation]:
        """
        Export property data from the SQL database.

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500}
        Returns
        -------
        output : list
                A list of rows represented as dictionaries.
        """
        with self.project.session as ses:
            subjects = parameters.pop('subjects', None)
            experiment = parameters.pop('experiment', None)

            query = ses.query(db.Computation)
            for key, val in parameters.items():
                if isinstance(val, str):
                    query = query.filter(
                        db.Computation.computation_attributes.any(name=key),
                        db.Computation.computation_attributes.any(str_value=val)
                    )
                else:
                    query = query.filter(
                        db.Computation.computation_attributes.any(name=key),
                        db.Computation.computation_attributes.any(value=val)
                    )
            if experiment is not None:
                query = query.filter(db.Computation.experiment.has(name=experiment))
            computations_all_subjects = query.all()

            # Filter out subjects, this is easier to do this way around than via SQL statements (feel free to rewrite!)
            computations = []
            if subjects is not None:
                for x in computations_all_subjects:
                    if set(x.subjects).issubset(subjects):
                        computations.append(x)
            else:
                computations = computations_all_subjects

            for computation in computations:
                _ = computation.data_dict

        return computations

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
    def temperature(self):
        """Get the temperature of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            temperature: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="temperature")
            temperature.value = value
            ses.commit()

    @property
    def time_step(self):
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            time_step: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="time_step")
            time_step.value = value
            ses.commit()

    @property
    def unit_system(self):
        """Get the unit_system of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            unit_system: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="unit_system")
            unit_system.str_value = value
            ses.commit()

    @property
    def number_of_configurations(self) -> int:
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            number_of_configurations: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment,
                                                                     name="number_of_configurations")
            number_of_configurations.value = value
            ses.commit()

    @property
    def number_of_atoms(self) -> int:
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            number_of_atoms: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment,
                                                            name="number_of_atoms")
            number_of_atoms.value = value
            ses.commit()

    @property
    def sample_rate(self):
        """Get the sample_rate of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            sample_rate: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="sample_rate")
            sample_rate.value = value
            ses.commit()

    @property
    def volume(self):
        """Get the volume of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            volume: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment, name="volume")
            volume.value = value
            ses.commit()

    @property
    def box_array(self):
        """Get the sample_rate of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
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
            experiment = get_or_create(ses, Experiment, name=self.name)
            for idx, box_length in enumerate(value):
                sample_rate: ExperimentData = get_or_create(
                    ses, ExperimentData, experiment=experiment, name=f"box_array_{idx}")
                sample_rate.value = box_length
            ses.commit()

    @property
    def species(self):
        species_dict = {}
        with self.project.session as ses:
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            for species in experiment.species:
                species_dict.update({
                    species.name: {
                        "indices": species.indices,
                        "mass": species.mass,
                        "charge": species.charge,
                        "particle_density": species.particle_density,
                        "molar_fraction": species.molar_fraction
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
                experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
                species = get_or_create(ses, Species, name=species_name, experiment=experiment)
                for species_attr, species_values in value[species_name].items():
                    try:
                        for idx, species_value in enumerate(species_values):
                            _ = get_or_create(
                                ses, SpeciesData,
                                name=species_attr, species=species, value=species_value
                            )
                    except TypeError:
                        # e.g., float or int values that are not iterable
                        if species_values is not None:
                            log.warning(f"Updating {species_attr} with {species_values}")
                            _ = get_or_create(
                                ses, SpeciesData,
                                name=species_attr, species=species, value=species_values
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
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
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
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
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
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
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
            experiment = ses.query(Experiment).filter(Experiment.name == self.name).first()
            get_or_create(ses, ExperimentData, name="read_file", str_value=value, experiment=experiment)
            ses.commit()

    @property
    def radial_distribution_function_state(self) -> bool:
        """Get the radial_distribution_function_state of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            rdf_state = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "radial_distribution_function_state").first()
        try:
            return rdf_state.value
        except AttributeError:
            return False

    @radial_distribution_function_state.setter
    def radial_distribution_function_state(self, value):
        """Set the radial_distribution_function_state of the experiment"""
        if value is None:
            return
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.name)
            rdf_state: ExperimentData = get_or_create(ses, ExperimentData, experiment=experiment,
                                                      name="radial_distribution_function_state")
            rdf_state.value = value
            ses.commit()
