from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()


class SystemProperty(Base):
    __tablename__ = 'system_properties'
    id = Column(Integer, primary_key=True)
    property = Column(String)
    analysis = Column(String)
    data_range = Column(Integer)

    data = relationship("Data", cascade="all, delete", back_populates="system_property")
    subjects = relationship("Subject", cascade="all, delete", back_populates="system_property")

    # TODO check that cascade is working properly!

    def __repr__(self):
        return f"{self.analysis} on {self.subjects}"

    def __init__(self, property, analysis, subjects, data_range, data):
        self.property = property
        self.analysis = analysis
        self.subjects = subjects
        self.data_range = data_range
        self.data = data


class Data(Base):
    __tablename__ = "data"

    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float, nullable=True)
    z = Column(Float, nullable=True)

    uncertainty = Column(Float, nullable=True)

    system_property_id = Column(Integer, ForeignKey('system_properties.id', ondelete="CASCADE"))

    system_property = relationship("SystemProperty", back_populates="data")

    def __repr__(self):
        return f"x:{self.x}"


class Subject(Base):
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True)
    subject = Column(String)

    system_property_id = Column(Integer, ForeignKey('system_properties.id', ondelete="CASCADE"))

    system_property = relationship("SystemProperty", back_populates="subjects")

    def __repr__(self):
        return f"{self.subject}"
