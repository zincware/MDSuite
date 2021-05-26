"""
Definition of the Database objects
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import DetachedInstanceError

import numpy as np

Base = declarative_base()


class SystemProperty(Base):
    """Main class for System properties

    Parameters
    ----------
    id: int, PK
    property: str
        Name of the Property
    analysis: str
        Name of the analysis
    data_range: int
        Data range
    information: str
        Additional information for the system property
    data: Data
        list of data associated with the system property (can be [x, y, z, uncertainty])
    subjects: Subject
        list of the subjects/species/molecules that are associated with the system property
    """
    __tablename__ = 'system_properties'
    id = Column(Integer, primary_key=True)
    property = Column(String)
    analysis = Column(String)
    data_range = Column(Integer)
    information = Column(String, nullable=True)

    data = relationship("Data", cascade="all, delete", back_populates="system_property")
    subjects = relationship("Subject", cascade="all, delete", back_populates="system_property")

    # TODO check that cascade is working properly!

    def __repr__(self):
        """System Property representation"""
        try:
            representation = f"{self.analysis} on {self.subjects}"
        except DetachedInstanceError:
            representation = f"{self.analysis}"

        if self.information is not None:
            representation += f" \t {self.information}"
        return representation

    def __init__(self, property, analysis, subjects, data_range, data, information=None):
        """System Property constructor"""
        self.property = property
        self.analysis = analysis
        self.subjects = subjects
        self.data_range = data_range
        self.data = data
        self.information = information

    def data_array(self) -> np.array:
        """Convert the one-to-many relationship data into a numpy array

        Returns
        -------
        np.array: all data converted to a numpy array with the shape (length, property) where a property contains the
        4 dimensions (x, y, z, uncertainty)

        """
        data = []
        for obj in self.data:
            data.append([obj.x, obj.y, obj.z, obj.uncertainty])
        return np.array(data)


class Data(Base):
    """Class for the data associated with SystemProperty

    Parameters
    ----------
    id: int, PK
    x: float, required
        x-value of the data
    y: float
        y-value of the data
    z: float
        z-value of the data
    uncertainty: float
        uncertainty value of the data
    """
    __tablename__ = "data"

    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float, nullable=True)
    z = Column(Float, nullable=True)

    uncertainty = Column(Float, nullable=True)

    system_property_id = Column(Integer, ForeignKey('system_properties.id', ondelete="CASCADE"))

    system_property = relationship("SystemProperty", back_populates="data")

    def __repr__(self):
        """Representation of the data"""
        return f"x:{self.x}"


class Subject(Base):
    """
    Class for the subjects associated with SystemProperty
    """
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True)
    subject = Column(String)

    system_property_id = Column(Integer, ForeignKey('system_properties.id', ondelete="CASCADE"))

    system_property = relationship("SystemProperty", back_populates="subjects")

    def __repr__(self):
        """Representation of the subject"""
        return f"{self.subject}"
