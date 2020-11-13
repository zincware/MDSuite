""" exceptions for the mdsuite program """


class NoElementInDump(Exception):
    """ Thrown when no elements are found in a dump file """
    pass


class NoTempInData(Exception):
    """ Thrown when no temperature is found in a data file """
    pass
