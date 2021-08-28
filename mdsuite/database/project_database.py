"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the project database.
"""
import logging
from .scheme import Project

from .database_base import DatabaseBase
from mdsuite.utils.database import get_or_create
from pathlib import Path

log = logging.getLogger(__file__)


class ProjectDatabase(DatabaseBase):
    """
    Class for the management of the project database.
    """

    def __init__(self):
        """
        Constructor for the Project database class.

        Parameters
        ----------
        name : str
                Path to the database location.
        """
        super().__init__(database_name="project.db")

    @property
    def project_id(self) -> int:
        """The id of this project in the database"""
        return 1

    @property
    def description(self):
        with self.session as ses:
            project = get_or_create(ses, Project, id=self.project_id)
            description = project.description
            ses.commit()

        return description

    @description.setter
    def description(self, value: str):
        """
        Allow users to add a short description to their project

        Parameters
        ----------
        value : str
                Description of the project. If the string ends in .txt, the contents of the txt file will be read. If
                it ends in .md, same outcome. Anything else will be read as is.
        """
        if Path(value).exists():
            value = Path(value).read_text()

        with self.session as ses:
            project = get_or_create(ses, Project, id=self.project_id)
            project.description = value
            ses.commit()

        # self.experiment_name = experiment_name
        #
        # self._experiment_id = None
    #
    # @property
    # def experiment(self) -> Experiment:
    #     """Write an entry for the Experiment the database
    #
    #     Returns
    #     -------
    #
    #     Experiment instance queried from the database
    #
    #     """
    #     if self._experiment_id is None:
    #         experiment = Experiment(name=self.experiment_name)
    #         with self.session as ses:
    #             ses.add(experiment)
    #             ses.commit()
    #             self._experiment_id = experiment.id
    #
    #     with self.session as ses:
    #         experiment = ses.query(Experiment).get(self._experiment_id)
    #     return experiment
