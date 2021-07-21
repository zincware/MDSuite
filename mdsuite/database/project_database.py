"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the project database.
"""
import logging
from .project_database_scheme import Experiment

from .database_base import DatabaseBase

log = logging.getLogger(__file__)


class ProjectDatabase(DatabaseBase):
    """
    Class for the management of the project database.
    """

    def __init__(self, name: str, experiment_name: str):
        """
        Constructor for the Project database class.

        Parameters
        ----------
        name : str
                Path to the database location.
        """
        super().__init__(name)
        self.experiment_name = experiment_name

        self._experiment_id = None

    @property
    def experiment(self) -> Experiment:
        """Write an entry for the Experiment the database

        Returns
        -------

        Experiment instance queried from the database

        """
        if self._experiment_id is None:
            experiment = Experiment(name=self.experiment_name)
            with self.session as ses:
                ses.add(experiment)
                ses.commit()
                self._experiment_id = experiment.id

        with self.session as ses:
            experiment = ses.query(Experiment).get(self._experiment_id)
        return experiment
