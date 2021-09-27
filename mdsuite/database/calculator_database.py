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
from sqlalchemy import and_

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
    data: dict = field(default_factory=dict)
    Subject: list[str] = field(default_factory=list)

@dataclass
class ComputationResults:
    data: dict = field(default_factory=dict)
    subjects: dict = field(default_factory=list)

class CalculatorDatabase:
    """Database Interactions of the calculator class

    This class handles the interaction of the calculator with the project database
    """

    def __init__(self, experiment):
        """Constructor for the calculator database"""
        self.experiment: Experiment = experiment
        self.db_computation: db.Computation = None
        self.database_group = None
        self.analysis_name = None
        self.load_data = None

        self._computation_data = []  # To be depreciated!
        self._queued_data = []

        # List of computation attributes that will be added to the database when it is written
        self.db_computation_attributes = []

    def clean_cache(self):
        """Clean the lists of computed data"""
        self._computation_data = []
        self._queued_data = []
        self.db_computation_attributes = []

    def prepare_db_entry(self):
        """Prepare a database entry based on the attributes defined in the init"""
        with self.experiment.project.session as ses:
            experiment = (
                ses.query(db.Experiment)
                    .filter(db.Experiment.name == self.experiment.name)
                    .first()
            )

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

        Returns
        -------
        db.Computation:
            Either a db.Computation object if the calculation was already performed or None

        """
        with self.experiment.project.session as ses:
            experiment = (
                ses.query(db.Experiment)
                    .filter(db.Experiment.name == self.experiment.name)
                    .first()
            )

            #  filter the correct experiment
            computations = ses.query(db.Computation).filter(
                db.Computation.experiment == experiment,
                db.Computation.name == self.analysis_name,
            )

            # filter the passed arguments and only run, if they changed
            for key, val in kwargs.items():
                computations = computations.filter(
                    db.Computation.computation_attributes.any(
                        and_(
                            db.ComputationAttribute.name == key,
                            db.ComputationAttribute.str_value == str(val),
                        )
                    )
                )

            # filter the version of the experiment, e.g. run new computation
            # if the experiment version has changed
            computations = computations.filter(
                db.Computation.computation_attributes.any(
                    and_(
                        db.ComputationAttribute.name == "version",
                        db.ComputationAttribute.value == self.experiment.version
                    )
                )
            )

            computations = computations.all()
            if len(computations) > 0:
                log.debug("Calculation already performed! Loading it up")
            # loading data_dict to avoid DetachedInstance errors
            # this can take some time, depending on the size of the data
            for computation in computations:
                _ = computation.data_dict
                _ = computation.data_range

        if len(computations) > 0:
            if len(computations) > 1:
                log.warning("Something went wrong! Found more than one computation with the given arguments!")
            return computations[0]  # it should only be one value
        else:
            for key, val in kwargs.items():
                computation_attribute = db.ComputationAttribute(
                    name=key, str_value=str(val)
                )

                self.db_computation_attributes.append(computation_attribute)

            # save the current experiment version in the ComputationAttributes
            experiment_version = db.ComputationAttribute(name="version", value=self.experiment.version)
            self.db_computation_attributes.append(experiment_version)

    def save_db_data(self, data=None):
        """Save all the collected computationattributes and computation data to the database

        This will be run after the computation was successful.
        """
        with self.experiment.project.session as ses:
            ses.add(self.db_computation)
            for val in self.db_computation_attributes:
                # I need to set the relation inside the session, otherwise it does not work.
                val.computation = self.db_computation
                ses.add(val)

            for data_obj in self._computation_data:
                log.warning("Depreciated computation data method")
                for data in data_obj.data:
                    for key, val in data.items():
                        computation_data = db.ComputationData(
                            computation=self.db_computation, value=val, dimension=key
                        )
                        ses.add(computation_data)
                        for subject in data_obj.Subject:
                            computation_subject = db.ComputationSpecies(
                                computation_data=computation_data, name=subject
                            )
                            ses.add(computation_subject)

            for data_obj in self._queued_data:
                # TODO consider renaming species to e.g., subjects, because species here can also be molecules
                data_obj: ComputationResults
                computation_result = db.ComputationResult(computation=self.db_computation, data=data_obj.data)
                species = ses.query(db.ExperimentSpecies).filter(db.ExperimentSpecies.name.in_(data_obj.subjects)).all()
                # in case of e.g. `System` species will be None
                if species is not None:
                    computation_result.species = species

                ses.add(computation_result)

            ses.commit()

    def queue_data(self, data, subjects):
        """Queue data to be stored in the database"""
        self._queued_data.append(
            ComputationResults(data=data, subjects=subjects)
        )

    def update_database(
            self, parameters: Union[dict, Parameters], delete_duplicate: bool = True
    ):
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
            self._computation_data.append(parameters)

        return

        if isinstance(parameters, Parameters):
            with self.experiment.project.session as ses:
                experiment = (
                    ses.query(db.Experiment)
                        .filter(db.Experiment.name == self.experiment.name)
                        .first()
                )
                computation = db.Computation(
                    experiment=experiment, name=parameters.Analysis
                )
                ses.add(computation)

                computation_property = db.ComputationAttribute(
                    computation=computation,
                    name="Property",
                    str_value=parameters.Property,
                )
                ses.add(computation_property)

                computation_analysis = db.ComputationAttribute(
                    computation=computation,
                    name="Analysis",
                    str_value=parameters.Analysis,
                )
                ses.add(computation_analysis)

                computation_data_range = db.ComputationAttribute(
                    computation=computation,
                    name="data_range",
                    value=parameters.data_range,
                )
                ses.add(computation_data_range)

                for subject in parameters.Subject:
                    computation_subject = db.ComputationAttribute(
                        computation=computation, name="subject", str_value=subject
                    )
                    ses.add(computation_subject)

                # data

                for data in parameters.data:
                    for key, val in data.items():
                        computation_data = db.ComputationData(
                            computation=computation, value=val, dimension=key
                        )
                        ses.add(computation_data)

                ses.commit()
        else:
            log.warning(
                "Using depreciated dictionary method - Please use dataclass instead."
            )
            # allow subjects and Subject
            if parameters.get("subjects") is None:
                try:
                    log.warning(
                        "Using depreciated `Subject` \t Please use `subjects` instead."
                    )
                    parameters["subjects"] = parameters["Subject"]
                except KeyError:
                    raise KeyError('Please add the key "subjects" to your calculator')

            with self.experiment.project.session as ses:
                experiment = (
                    ses.query(db.Experiment)
                        .filter(db.Experiment.name == self.experiment.name)
                        .first()
                )
                computation = db.Computation(
                    experiment=experiment, name=parameters["Analysis"]
                )
                ses.add(computation)

                computation_property = db.ComputationAttribute(
                    computation=computation,
                    name="Property",
                    str_value=parameters["Property"],
                )
                ses.add(computation_property)

                computation_analysis = db.ComputationAttribute(
                    computation=computation,
                    name="Analysis",
                    str_value=parameters["Analysis"],
                )
                ses.add(computation_analysis)

                computation_data_range = db.ComputationAttribute(
                    computation=computation,
                    name="data_range",
                    value=parameters["data_range"],
                )
                ses.add(computation_data_range)

                for subject in parameters["subjects"]:
                    computation_subject = db.ComputationAttribute(
                        computation=computation, name="subject", str_value=subject
                    )
                    ses.add(computation_subject)

                # data

                for data in parameters["data"]:
                    for key, val in data.items():
                        computation_data = db.ComputationData(
                            computation=computation, value=val, dimension=key
                        )
                        ses.add(computation_data)

                ses.commit()

    ##### REMOVE ######
    # TODO rename and potentially move to a RDF based parent class
    def _get_rdf_data(self) -> List[db.Computation]:
        """Fill the data_files list with filenames of the rdf tensor_values"""
        # TODO replace with exp.load.RDF()
        with self.experiment.project.session as ses:
            computations = (
                ses.query(db.Computation)
                    .filter(
                    db.Computation.computation_attributes.any(
                        str_value="Radial_Distribution_Function", name="Property"
                    )
                )
                    .all()
            )

            for computation in computations:
                _ = computation.data_dict
                _ = computation.data_range

        return computations

    # TODO rename and potentially move to a RDF based parent class
    def _load_rdf_from_file(self, computation: db.Computation):
        """Load the raw rdf tensor_values from a directory"""

        self.radii = np.array(computation.data_dict["x"]).astype(float)[1:]
        self.rdf = np.array(computation.data_dict["y"]).astype(float)[1:]
#####################
