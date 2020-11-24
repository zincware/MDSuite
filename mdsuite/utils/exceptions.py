""" exceptions for the mdsuite program """


class NoElementInDump(Exception):
    """ Thrown when no elements are found in a dump file """
    pass


class NoTempInData(Exception):
    """ Thrown when no temperature is found in a data file """
    pass

class NotApplicableToAnalysis(Exception):
    """ Thrown when the function is not applicable to the type of analysis being performed """

    print("This particular function does not apply to the analysis being performed and so is not implemented")

    pass
