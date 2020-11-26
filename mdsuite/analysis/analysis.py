""" Parent class for different analysis """

import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    """ Parent class for analysis modules """

    def __init__(self, obj, plot=True, save=True, data_range=500, x_label=None, y_label=None, analysis_name=None):
        """ Python constructor """

        self.parent = obj  # Experiment object to get properties from
        self.data_range = data_range  # Data range over which to evaluate
        self.plot = plot  # Whether or not to plot the data and save a figure
        self.save = save  # Whether or not to save the calculated data (Default is true)

        self.x_label = x_label  # x label of the figure
        self.y_label = y_label  # y label of the figure
        self.analysis_name = analysis_name  # what to save the figure as

    def _autocorrelation_time(self):
        """ get the autocorrelation time for the relevant property to ensure good error sampling """
        raise NotImplementedError  # Implemented in the child class

    def _save_data(self, title, data):
        """ Save data to the save data directory """

        np.save(f"{self.parent.storage_path}/{self.parent.analysis_name}/data/{title}.npy", data)

    def _plot_data(self):
        """ Plot the data generated during the analysis """

        plt.xlabel(rf'{self.x_label}')  # set the x label
        plt.ylabel(rf'{self.y_label}')  # set the y label
        plt.legend()  # enable the legend
        plt.savefig(f"{self.parent.storage_path}/{self.parent.analysis_name}/Figures/{self.analysis_name}.eps",
                    dpi=600, format='eps')

    def run_analysis(self):
        """ Run the appropriate analysis


        Should follow the general outline detailed below:

        self._autocorrelation_time()  # Calculate the relevent autocorrelation time
        self._analysis()  # Can be diffusion coefficients or whatever is being calculated, but run the calculation
        self._error_anaysis  # Run an error analysis, could be done during the calculation, or may have to be for the
                                sake of memory.
        self._update_experiment  # Update the main experiment class with the calculated properties
        """
        raise NotImplementedError  # Implement in the child class





