"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Access the computations database
"""

from .database_base import DatabaseBase
from .project_database_scheme import Computation, Experiment, ComputationAttribute, ComputationData

from typing import List


class ComputationDatabase(DatabaseBase):

    def __init__(self, name: str, experiment: Experiment):
        super().__init__(name)

        self.experiment = experiment
        self._computation_id = None

    def set_attributes(self, name, value=None, str_value=None):
        with self.session as ses:
            computation_attribute = ComputationAttribute(name=name,
                                                         value=value,
                                                         str_value=str_value,
                                                         computation=self.computation)
            ses.add(computation_attribute)
            ses.commit()

    def set_data(self, value: float, dimension: str):
        with self.session as ses:
            computation_data = ComputationData(value=value,
                                               dimension=dimension,
                                               computation=self.computation)
            ses.add(computation_data)
            ses.commit()

    @property
    def computation(self):
        """Write an entry for the computation the database

        This sets the default id of the instance and writes it to the database

        Returns
        -------
        computation instance queried from the database
        """
        if self._computation_id is None:
            computation = Computation(experiment=self.experiment)
            with self.session as ses:
                ses.add(computation)
                ses.commit()
                self._computation_id = computation.id

        with self.session as ses:
            computation = ses.query(Computation).get(self._computation_id)
        return computation

    @property
    def attributes(self) -> List[ComputationAttribute]:
        """Get the attributes of this instances computation

        Returns
        -------
            list, A List of ComputationAttributes for this instances computation
        """
        with self.session as ses:
            out = ses.query(ComputationAttribute).filter_by(computation_id=self._computation_id).all()

        return out

    @attributes.setter
    def attributes(self, value: dict):
        """Write attributes

        Parameters
        ----------
        value: dict
            A dictionary {name, value, str_value} to be written to the database
        """
        with self.session as ses:
            computation_attribute = ComputationAttribute(**value,
                                                         computation=self.computation)
            ses.add(computation_attribute)
            ses.commit()

    @property
    def data(self):
        """Get the data of this instances computation

        Returns
        -------
            list, A List of ComputationData for this instances computation
        """
        with self.session as ses:
            out = ses.query(ComputationData).filter_by(computation_id=self._computation_id).all()
        return out

    @data.setter
    def data(self, value):
        """Write data

        Parameters
        ----------
        value: dict
            A dictionary {value, uncertainty, dimension} to be written to the database
        """
        with self.session as ses:
            data = ComputationData(**value, computation=self.computation)
            ses.add(data)
            ses.commit()
