"""
Class for the calculation of the coordinated numbers
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import h5py as hf

# MDSuite imports
from mdsuite.utils.exceptions import *
from mdsuite.calculators.calculator import Calculator


class KirkwoodBuffIntegral(Calculator):
    """
    Class for the calculation of the Kikrwood-Buff integrals

    Attributes
    ----------
    experiment : class object
                        Class object of the experiment.
    plot : bool (default=True)
                        Decision to plot the analysis.
    save : bool (default=True)
                        Decision to save the generated tensor_values arrays.

    data_range : int (default=500)
                        Range over which the property should be evaluated. This is not applicable to the current
                        analysis as the full rdf will be calculated.
    x_label : str
                        How to label the x axis of the saved plot.
    y_label : str
                        How to label the y axis of the saved plot.
    analysis_name : str
                        Name of the analysis. used in saving of the tensor_values and figure.
    file_to_study : str
                        The tensor_values file corresponding to the rdf being studied.
    data_directory : str
                        The directory in which to find this tensor_values.
    data_files : list
                        list of files to be analyzed.
    rdf = None : list
                        rdf tensor_values being studied.
    radii = None : list
                        radii tensor_values corresponding to the rdf.
    species_tuple : list
                        A list of species combinations being studied.
    pomf : list
                        List of tensor_values of the potential of mean-force for the current analysis.
    """

    def __init__(self, experiment, plot=True, save=True, data_range=None, x_label=r'r ($\AA$)', y_label=r'$G(\mathbf{r})$',
                 analysis_name='Kirkwood-Buff_Integral'):
        """
        Python constructor for the class

        Parameters
        ----------
        experiment : class object
                        Class object of the experiment.
        plot : bool (default=True)
                            Decision to plot the analysis.
        save : bool (default=True)
                            Decision to save the generated tensor_values arrays.

        data_range : int (default=500)
                            Range over which the property should be evaluated. This is not applicable to the current
                            analysis as the full rdf will be calculated.
        x_label : str
                            How to label the x axis of the saved plot.
        y_label : str
                            How to label the y axis of the saved plot.
        analysis_name : str
                            Name of the analysis. used in saving of the tensor_values and figure.
        """

        super().__init__(experiment, plot, save, data_range, x_label, y_label, analysis_name)
        self.file_to_study = None                                             # RDF file being studied
        self.data_directory = f'{experiment.storage_path}/{experiment.analysis_name}/tensor_values'  # directory in which tensor_values is stored
        self.data_files = []                                                  # array of the files in tensor_values directory
        self.rdf = None                                                       # rdf being studied
        self.radii = None                                                     # radii of the rdf
        self.species_tuple = None                                             # Which species are being studied
        self.kb_integral = None                                               # Kirkwood-Buff integral for the rdf
        self.database_group = 'kirkwood_buff_integral'                        # Which database_path group to save the tensor_values in


    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _get_rdf_data(self):
        """
        Fill the data_files list with filenames of the rdf tensor_values
        """
        with hf.File(os.path.join(self.experiment.database_path, 'analysis_data.hdf5'), 'r') as db:
            for item in db['radial_distribution_function']:  # loop over the files
                self.data_files.append(item)  # Append to the data_file attribute

    def _load_rdf_from_file(self):
        """
        Load the raw rdf tensor_values from a directory
        """

        with hf.File(os.path.join(self.experiment.database_path, 'analysis_data.hdf5'), 'r') as db:
            self.radii, self.rdf = db['radial_distribution_function'][self.file_to_study]

    def _calculate_kb_integral(self):
        """
        calculate the Kirkwood-Buff integral
        """

        self.kb_integral = []  # empty the integration tensor_values

        for i in range(1, len(self.radii)):
            self.kb_integral.append(4*np.pi*(np.trapz((self.rdf[1:i] - 1)*(self.radii[1:i])**2, x=self.radii[1:i])))

    def run_analysis(self):
        """
        Calculate the potential of mean-force and perform error analysis
        """

        self._get_rdf_data()  # fill the tensor_values array with tensor_values

        for data in self.data_files:
            self.file_to_study = data        # Set the file to study
            self.species_tuple = data[:-33]  # set the tuple
            self._load_rdf_from_file()       # Load the rdf tensor_values for the set file
            self._calculate_kb_integral()    # Integrate the rdf and calculate the KB integral

            # Save if necessary
            if self.save:
                self._save_data(f"{self.analysis_name}_{self.species_tuple}", [self.radii[1:], self.kb_integral])
            # Plot if necessary
            if self.plot:
                plt.plot(self.radii[1:], self.kb_integral, label=f"{self.species_tuple}")
                self._plot_data(title=f"{self.analysis_name}_{self.species_tuple}")

