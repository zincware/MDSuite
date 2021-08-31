"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: database management for the calculator class
"""
from __future__ import annotations

import mdsuite.database.scheme as db
from mdsuite.utils.database import get_or_create

import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment

log = logging.getLogger(__file__)


class CalculatorDatabase:
    def __init__(self, experiment):
        self.experiment: Experiment = experiment

    def update_database(self, parameters: dict, delete_duplicate: bool = True):
        """
        Add data to the database

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500, "data": 1.8e-9}
        delete_duplicate : bool
                If true, duplicate entries will be deleted.
        Returns
        -------
        Updates the sql database
        """
        # allow subjects and Subject
        if parameters.get("subjects") is None:
            try:
                log.warning("Using depreciated `Subject` \t Please use `subjects` instead.")
                parameters['subjects'] = parameters['Subject']
            except KeyError:
                raise KeyError('Please add the key "subjects" to your calculator')

        with self.experiment.project.session as ses:
            experiment = ses.query(db.Experiment).filter(db.Experiment.name == self.experiment.experiment_name).first()
            computation = db.Computation(experiment=experiment)
            ses.add(computation)

            computation_property = db.ComputationAttribute(computation=computation, name="Property",
                                                           str_value=parameters['Property'])
            ses.add(computation_property)

            computation_analysis = db.ComputationAttribute(computation=computation, name="Analysis",
                                                           str_value=parameters['Analysis'])
            ses.add(computation_analysis)

            computation_data_range = db.ComputationAttribute(computation=computation, name="data_range",
                                                             value=parameters['data_range'])
            ses.add(computation_data_range)

            for subject in parameters['subjects']:
                computation_subject = db.ComputationAttribute(computation=computation, name="subject",
                                                              str_value=subject)
                ses.add(computation_subject)

            # data

            for data in parameters['data']:
                for key, val in data.items():
                    computation_data = db.ComputationData(computation=computation, value=val, dimension=key)
                    ses.add(computation_data)

            ses.commit()

    # TODO rename and potentially move to a RDF based parent class
    def _get_rdf_data(self):
        """Fill the data_files list with filenames of the rdf tensor_values"""
        with self.experiment.project.session as ses:
            computations = ses.query(db.Computation).filter(
                db.Computation.computation_attributes.any(str_value="RDF", name="Property")
            ).all()

            for computation in computations:
                _ = computation.data_dict
                _ = computation.data_range

        return computations

    # TODO rename and potentially move to a RDF based parent class
    def _load_rdf_from_file(self, computation: db.Computation):
        """Load the raw rdf tensor_values from a directory"""

        self.radii = np.array(computation.data_dict['x']).astype(float)[1:]
        self.rdf = np.array(computation.data_dict['y']).astype(float)[1:]

    # @property
    # def data(self) -> np.ndarray:
    #     """
    #
    #     Returns
    #     -------
    #     object that contains `
    #
    #     """
    #     with self.experiment.project.session as ses:
    #         ses.query(db.Computation)
    #
    #     return 0
        #
        # # if delete_duplicate:
        # #     self._delete_duplicate_rows(parameters)
        # # else:
        # #     if self._check_row_existence(parameters):
        # #         log.info("Note, an entry with these parameters already exists in the database.")
        #
        # log.debug(f'Adding {parameters.get("Property")} to database!')
        #
        # with self.experiment.project.session as ses:
        #     # Create a Data instance to store the value
        #     # TODO use **parameters instead with parameters.pop
        #
        #     try:  # check if it is a list
        #         data = [Data(**param) for param in parameters['data']]  # param is a dict
        #     except TypeError:
        #         try:  # check if it is a dictionary with keys [x, y, z, uncertainty]
        #             data = [Data(parameters['data'])]
        #         except TypeError:
        #             data = [Data(x=parameters['data'])]
        #
        #     try:
        #         subjects = [Subject(subject=param) for param in parameters['subjects']]  # param is a string
        #     except TypeError:
        #         subjects = [Subject(subject=parameters['subjects'])]  # param is a string
        #
        #     log.debug(f"Subjects are: {subjects}")
        #
        #     # Create s SystemProperty instance to store the values
        #     system_property = SystemProperty(
        #         property=parameters['Property'],
        #         analysis=parameters['Analysis'],
        #         data_range=parameters['data_range'],
        #         subjects=subjects,
        #         data=data,
        #         information=parameters.get("information")
        #     )
        #
        #     log.debug(f"Created: {system_property}")
        #
        #     # add to the session
        #     ses.add(system_property)
        #
        #     # commit to the database
        #     ses.commit()
        #
        # log.debug("Values successfully written to database. Closed database session.")
