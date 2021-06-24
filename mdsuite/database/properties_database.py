"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Python module for the properties database.
"""
import logging
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.session import Session
from .database_scheme import Base, SystemProperty, Data, Subject

log = logging.getLogger(__file__)


class PropertiesDatabase:
    """
    A class to control the properties database.
    """

    def __init__(self, name: str):
        """
        Constructor for the PropertiesDatabase class.

        Parameters
        ----------
        name : str
                Name of the database. Should be the full path to the name.
        """
        self.name = name
        self.engine = sa.create_engine(f"sqlite+pysqlite:///{self.name}", echo=False, future=True)

        # self.engine = sa.create_engine(f"sqlite:///:memory:", echo=True)
        self.Session: sessionmaker

        self.Base = Base

        self.get_session()
        self.build_database()

    def get_session(self):
        """Create a session"""
        log.debug('Creating sessionmaker')
        self.Session = sessionmaker(bind=self.engine, future=True)

    def build_database(self):
        """Create the database scheme"""
        log.debug('Creating Database if not existing')
        self.Base.metadata.create_all(self.engine)

    @staticmethod
    def _build_subject_query(subjects, query, ses):
        """Query the one->many subjects relationship

        Parameters
        ----------
        subjects: list
            List of the subjects to query
        query:
            The query objects
        ses:
            session object
        """

        for subject in subjects:
            query = query.filter(SystemProperty.subjects.any(subject=subject))
        # select the samples where the subject conditions are full filled and connect them with "and" (multiple filters)

        subject_objs = ses.query(Subject.id).filter(Subject.subject.in_(subjects)).distinct()
        # get all subjects where the subject is in the given list

        query = query.filter(~SystemProperty.subjects.any(
            Subject.id.notin_(subject_objs)  # remove all, that have additional subjects in their query
        ))

        return query

    def _check_row_existence(self, parameters: dict):
        """
        Check if a row exists.

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500, "data": 1.8e-9}
        Returns
        -------
        result : bool
                True or False depending on existence.
        """
        log.debug(f'Check if row for {parameters.keys()} exists')
        with self.Session() as ses:
            ses: Session

            query = ses.query(SystemProperty)

            if parameters.get("data_range") is not None:
                query = query.filter_by(data_range=parameters['data_range'])
            if parameters.get("Analysis") is not None:
                query = query.filter_by(analysis=parameters['Analysis'])
            if parameters.get("information") is not None:
                query = query.filter_by(information=parameters['information'])

            log.debug(f"check, without subjects: {query.all()}")

            if parameters.get('subjects') is not None:
                query = self._build_subject_query(parameters['subjects'], query, ses)
            query = query.all()

        log.debug(f'Check yielded {query}')
        return len(query) > 0

    def _delete_duplicate_rows(self, parameters: dict):
        """
        Delete duplicate rows

        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500, "data": 1.8e-9}
        Returns
        -------
        result : bool
                True or False depending on existence.
        """

        log.debug(f"Parameters: {parameters.keys()}")

        with self.Session() as ses:
            ses: Session

            query = ses.query(SystemProperty).filter_by(
                data_range=parameters['data_range'],
                analysis=parameters['Analysis'],
            )
            if parameters.get("information") is not None:
                query = query.filter_by(information=parameters.get("information"))

            if parameters.get('subjects') is not None:
                query = self._build_subject_query(parameters['subjects'], query, ses)

            system_properties = query.all()

            log.debug(f'Removing {system_properties} from database')
            for system_property in system_properties:
                ses.delete(system_property)

            ses.commit()

    def add_data(self, parameters: dict, delete_duplicate: bool = True):
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
                parameters['subjects'] = parameters['Subject']
            except KeyError:
                raise KeyError('Please add the key "subjects" to your calculator')

        if delete_duplicate:
            self._delete_duplicate_rows(parameters)
        else:
            if self._check_row_existence(parameters):
                log.info("Note, an entry with these parameters already exists in the database.")

        log.debug(f'Adding {parameters.get("Property")} to database!')

        with self.Session() as ses:
            ses: Session

            # Create a Data instance to store the value
            # TODO use **parameters instead with parameters.pop

            try:  # check if it is a list
                data = [Data(**param) for param in parameters['data']]  # param is a dict
            except TypeError:
                try:  # check if it is a dictionary with keys [x, y, z, uncertainty]
                    data = [Data(parameters['data'])]
                except TypeError:
                    data = [Data(x=parameters['data'])]

            try:
                subjects = [Subject(subject=param) for param in parameters['subjects']]  # param is a string
            except TypeError:
                subjects = [Subject(subject=parameters['subjects'])]  # param is a string

            log.debug(f"Subjects are: {subjects}")

            # Create s SystemProperty instance to store the values
            system_property = SystemProperty(
                property=parameters['Property'],
                analysis=parameters['Analysis'],
                data_range=parameters['data_range'],
                subjects=subjects,
                data=data,
                information=parameters.get("information")
            )

            log.debug(f"Created: {system_property}")

            # add to the session
            ses.add(system_property)

            # commit to the database
            ses.commit()

        log.debug("Values successfully written to database. Closed database session.")

    def load_data(self, parameters: dict) -> list:
        """
        Load some data from the database.
        Parameters
        ----------
        parameters : dict
                Parameters to be used in the addition, i.e.
                .. code-block::

                   {"Analysis": "Green_Kubo_Self_Diffusion", "Subject": "Na", "data_range": 500}

        Returns
        -------
        output : list
                All rows matching the parameters represented as a dictionary.
        """

        log.debug(f'querying {parameters} from database')

        with self.Session() as ses:
            ses: Session

            subjects = parameters.pop('subjects', None)
            query = ses.query(SystemProperty).filter_by(**parameters)

            if subjects is not None:
                query = self._build_subject_query(subjects, query, ses)

            system_properties = query.all()

            # Iterate over data so that the information gets pulled from the database
            # Note: If you keep the session open, this would not be necessary
            for system_property in system_properties:
                _ = system_property.data
                _ = system_property.subjects

        return system_properties
