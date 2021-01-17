"""
Parent class for MDSuite transformations
"""


class Transformations:
    """
    Parent class for MDSuite transformations.
    """

    def __init__(self):
        """
        Constuctor for the parent class
        """
        pass

    def run_transformation(self):
        """
        Perform the transformation
        """
        raise NotImplementedError  # implemented in child class.
