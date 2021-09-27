"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Collection of custom database types

References
----------

https://docs.sqlalchemy.org/en/14/orm/extensions/mutable.html
"""

from sqlalchemy.types import TypeDecorator, VARCHAR
from sqlalchemy.ext.mutable import Mutable

import json


class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string."""

    impl = VARCHAR

    def process_bind_param(self, value: dict, dialect):
        """Provide a bound value processing function

        Convert a dictionary to a json string and store the string in the database
        """
        if value is not None:
            value = json.dumps(value)
        return value

    def process_result_value(self, value, dialect) -> dict:
        """Receive a result-row column value to be converted

        Convert a loaded string from the database into a dict object
        """
        if value is not None:
            value = json.loads(value)
        return value


class MutableDict(Mutable, dict):
    """Subclassed version of a dictionary used in the database"""
    @classmethod
    def coerce(cls, key, value):
        """Convert plain dictionaries to MutableDict."""

        if not isinstance(value, MutableDict):
            if isinstance(value, dict):
                return MutableDict(value)

            # this call will raise ValueError
            return Mutable.coerce(key, value)
        else:
            return value

    def __setitem__(self, key, value):
        """Detect dictionary set events and emit change events."""

        dict.__setitem__(self, key, value)
        self.changed()

    def __delitem__(self, key):
        """Detect dictionary del events and emit change events."""

        dict.__delitem__(self, key)
        self.changed()
