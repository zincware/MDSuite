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
from mdsuite.database.scheme import Project, Experiment, ExperimentData
from mdsuite.utils.database import get_or_create

log = logging.getLogger(__file__)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Project


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
    def number_of_configurations(self):
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            number_of_configurations = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "number_of_configurations").first()
        try:
            return number_of_configurations.value
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
    def number_of_atoms(self):
        """Get the time_step of the experiment"""
        with self.project.session as ses:
            experiment = get_or_create(ses, Experiment, name=self.experiment_name)
            number_of_atoms = ses.query(ExperimentData).filter(ExperimentData.experiment == experiment).filter(
                ExperimentData.name == "number_of_atoms").first()
        try:
            return number_of_atoms.value
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
