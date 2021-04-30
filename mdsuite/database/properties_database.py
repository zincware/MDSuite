"""
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
        log.debug('Creating sessionmaker')
        self.Session = sessionmaker(bind=self.engine, future=True)

    def build_database(self):
        log.debug('Creating Database if not existing')
        self.Base.metadata.create_all(self.engine)

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
        log.debug(f'Check if row for {parameters} exists')
        with self.Session() as ses:
            ses: Session

            # TODO use **parameters instead
            # TODO does not work!
            query = ses.query(SystemProperty).filter_by(
                subject=parameters['Subject'], data_range=parameters['data_range']).all()

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
        # return None

        log.debug(f"Parameters: {parameters.get('subjects')}")

        with self.Session() as ses:
            ses: Session

            query = ses.query(SystemProperty).filter_by(
                data_range=parameters['data_range'],
                analysis=parameters['Analysis'])

            # assume subjects are only two values
            if parameters['subjects'][0] == parameters['subjects'][1]:
                subject = parameters['subjects'][0]
                query = query.filter(SystemProperty.subjects.any(subject=subject))
                # filter by subject in
                query = query.filter(~SystemProperty.subjects.any(Subject.subject != subject))
                # filter by not in (not subject)
            else:
                for subject in parameters['subjects']:
                    query = query.filter(SystemProperty.subjects.any(subject=subject))

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
        if delete_duplicate:
            self._delete_duplicate_rows(parameters)
        else:
            if self._check_row_existence(parameters):
                print("Note, an entry with these parameters already exists in the database.")

        log.debug(f'Adding {parameters.get("Property")} to database!')

        with self.Session() as ses:
            ses: Session

            # Create a Data instance to store the value
            # TODO use **parameters instead with parameters.pop

            data = [Data(**param) for param in parameters['data']]  # param is a dict
            subjects = [Subject(subject=param) for param in parameters['subjects']]  # param is a string
            log.debug(f"Subjects are: {subjects}")
            # try:
            #     # self.log.debug(f"Constructing data objects from {len(parameters['data'])} passed values")
            #     data = []
            #     for data_point in parameters['data']:
            #         data.append(Data(**data_point))
            # except TypeError:
            #     # self.log.debug(f"Constructing data objects from {parameters['data']}")
            #     data.append(Data(x=parameters['data'], uncertainty=parameters.get('uncertainty')))

            # Create s SystemProperty instance to store the values

            system_property = SystemProperty(
                property=parameters['Property'],
                analysis=parameters['Analysis'],
                data_range=parameters['data_range'],
                subjects=subjects,
                data=data)

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
                {"Analysis": "Green_Kubo_Self_Diffusion",
                 "Subject": "Na",
                 "data_range": 500}

        Returns
        -------
        output : list
                All rows matching the parameters represented as a dictionary.
        """

        log.debug(f'querying {parameters} from database')

        with self.Session() as ses:
            ses: Session

            system_properties = ses.query(SystemProperty).filter_by(**parameters).all()

            # Iterate over data so that the information gets pulled from the database
            # Note: If you keep the session open, this would not be necessary
            for system_property in system_properties:
                _ = system_property.data
                _ = system_property.subjects

        return system_properties
