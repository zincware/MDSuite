"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description:
"""
from dataclasses import make_dataclass


def generate_dataclass(**kwargs):
    """
    Generate a dataclass from the parsed kwargs.

    This method allows a user to create a dataclass from a set of keyword arguments.
    This is used mainly in the calculators to create dataclasses for the SQL database.

    Parameters
    ----------
    kwargs : Any
            Attributes that you wish to have turned into a dataclass.

    Returns
    -------
    dataclass : object
            A dataclass with the attributes passed to the function.
    """
    attributes = [(item, type(value)) for item, value in kwargs.items()]
    class_maker = make_dataclass("Stored_Parameters", attributes)

    return class_maker(**kwargs)


class NoneType:
    """Custom NoneType that is != None

    Examples
    --------
    you can check for NoneType or None
    >>> x = NoneType
    >>> x is NoneType
    >>> x is not None
    """

    def __init__(self):
        raise NotImplementedError()
