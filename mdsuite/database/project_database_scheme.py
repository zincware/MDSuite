"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Module for the project database schema.
"""
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import DetachedInstanceError

import numpy as np

Base = declarative_base()


class Project(Base):
    """
    Class for the project table.
    """
    __table_name__ = 'project'
    id = Column(Integer, primary_key=True)
    name = Column(String)

    def __init__(self):
        """
        Constructor for the Project class.
        """
        pass



