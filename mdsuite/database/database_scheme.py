from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship

# self.table = table('system_properties', column('Property'), column('Analysis'),
#                            column('Subject'), column('data_range'),
#                            column('data'), column('uncertainty'))

# stmt = sql.text("CREATE TABLE system_properties (Property varchar(255), Analysis varchar(255), "
#                             "Subject varchar(255), data_range INT, data REAL , uncertainty REAL)")

Base = declarative_base()


class SystemProperty(Base):
    __tablename__ = 'system_properties'
    id = Column(Integer, primary_key=True)
    property = Column(String)
    analysis = Column(String)
    subject = Column(String)
    data_range = Column(Integer)

    data = relationship("Data", cascade="all, delete", back_populates="system_property")
    # TODO check that cascade is working properly!

    def __repr__(self):
        return f"This is a {self.property} - {self.analysis}"

    def __init__(self, property, analysis, subject, data_range, data):
        self.property = property
        self.analysis = analysis
        self.subject = subject
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
