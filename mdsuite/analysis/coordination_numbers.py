""" Class for the calculation of the coordinated numbers """

from mdsuite.utils.exceptions import *
from mdsuite.analysis.analysis import Analysis

class _CoordinationNumbers(Analysis):
    """ Class for the calculation of coordination numbers """

    def __init__(self, obj, plot=True, save=True, data_range=None, x_label='r ($\AA$)', y_label='CN', analysis_name='Coordination_Numbers'):
        """ Python constructor """
        super().__init__(obj, plot, save, data_range, x_label, y_label, analysis_name)

        # Calculate the rdf if it has not been done already
        if self.parent.radial_distribution_function_state is False:
            self.parent.radial_distribution_function()

    def _autocorrelation_time(self):
        """ Not needed in this analysis """
        raise NotApplicableToAnalysis

    def _integrate_rdf(self):
        """ Integrate the rdf currently in the class state """
        raise NotImplementedError

    def _find_minimums(self):
        """ Use min finding algorithm to determine the minimums of the function """
        raise NotImplementedError

    def run_analysis(self):
        """ Calculate the coordination numbers and perform error analysis
        """
        """
        for rdf in rdf_list:
            self._integrte_rdf()  # integrate the rdf
            self._find_minimums()  # determine the CN's at the minimums
            self._update_parent_class()  # update the parent class with the new information
        """