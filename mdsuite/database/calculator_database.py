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
from dataclasses import dataclass, field

import logging
import numpy as np
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment

log = logging.getLogger(__name__)


@dataclass
class Parameters:
    Property: str
    Analysis: str
    data_range: int
    data: list[dict] = field(default_factory=list)
    Subject: list[str] = field(default_factory=list)


class CalculatorDatabase:
    def __init__(self, experiment):
        self.experiment: Experiment = experiment
        self.db_computation: db.Computation = None
        self.database_group = None
        self.analysis_name = None

        # List of computation attributes that will be added to the database when it is written
        self.db_computation_attributes = []

    def prepare_db_entry(self):
        """Prepare a database entry based on the attributes defined in the init"""
        with self.experiment.project.session as ses:
            experiment = ses.query(db.Experiment).filter(db.Experiment.name == self.experiment.name).first()

        self.db_computation = db.Computation(experiment=experiment)
        self.db_computation.name = self.analysis_name

    def update_db_entry_with_kwargs(self, **kwargs):
        """Update the database entry with the given user args/kwargs

        Parameters
        ----------
        kwargs: all arguments that are passed to the call method and should be stored in the database


        Notes
        -----
        This does require kwargs, args do not work!

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

            computation_attribute = db.ComputationAttribute(
                computation=self.db_computation,
                name=key,
                str_value=str(val)
            )

            self.db_computation_attributes.append(computation_attribute)

    def save_db_data(self, data=None):
        with self.experiment.project.session as ses:
            ses.add(self.db_computation)
            for val in self.db_computation_attributes:
                ses.add(val)

            ses.commit()

    def update_database(self, parameters: Union[dict, Parameters], delete_duplicate: bool = True):
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
        if isinstance(parameters, Parameters):
            with self.experiment.project.session as ses:
                experiment = ses.query(db.Experiment).filter(db.Experiment.name == self.experiment.name).first()
                computation = db.Computation(experiment=experiment, name=parameters.Analysis)
                ses.add(computation)

                computation_property = db.ComputationAttribute(computation=computation, name="Property",
                                                               str_value=parameters.Property)
                ses.add(computation_property)

                computation_analysis = db.ComputationAttribute(computation=computation, name="Analysis",
                                                               str_value=parameters.Analysis)
                ses.add(computation_analysis)

                computation_data_range = db.ComputationAttribute(computation=computation, name="data_range",
                                                                 value=parameters.data_range)
                ses.add(computation_data_range)

                for subject in parameters.Subject:
                    computation_subject = db.ComputationAttribute(computation=computation, name="subject",
                                                                  str_value=subject)
                    ses.add(computation_subject)

                # data

                for data in parameters.data:
                    for key, val in data.items():
                        computation_data = db.ComputationData(computation=computation, value=val, dimension=key)
                        ses.add(computation_data)

                ses.commit()
        else:
            log.warning("Using depreciated dictionary method - Please use dataclass instead.")
            # allow subjects and Subject
            if parameters.get("subjects") is None:
                try:
                    log.warning("Using depreciated `Subject` \t Please use `subjects` instead.")
                    parameters['subjects'] = parameters['Subject']
                except KeyError:
                    raise KeyError('Please add the key "subjects" to your calculator')

            with self.experiment.project.session as ses:
                experiment = ses.query(db.Experiment).filter(db.Experiment.name == self.experiment.name).first()
                computation = db.Computation(experiment=experiment, name=parameters["Analysis"])
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
    def _get_rdf_data(self) -> List[db.Computation]:
        """Fill the data_files list with filenames of the rdf tensor_values"""
        # TODO replace with exp.load.RDF()
        with self.experiment.project.session as ses:
            computations = ses.query(db.Computation).filter(
                db.Computation.computation_attributes.any(str_value="Radial_Distribution_Function", name="Property")
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
