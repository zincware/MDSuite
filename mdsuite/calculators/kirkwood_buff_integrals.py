"""
Class for the calculation of the coordinated numbers
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Union
from mdsuite.database.analysis_database import AnalysisDatabase

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

    def __init__(self, experiment):
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

        super().__init__(experiment)
        self.file_to_study = None
        self.data_files = []
        self.rdf = None
        self.radii = None
        self.species_tuple = None
        self.kb_integral = None
        self.database_group = 'Kirkwood_Buff_Integral'
        self.x_label = r'r ($\AA$)'
        self.y_label = r'$G(\mathbf{r})$'
        self.analysis_name = 'Kirkwood-Buff_Integral'

        self.post_generation = True

    def __call__(self, plot=True, save=True, data_range=1, export: bool = False):
        self.update_user_args(plot=plot, save=save, data_range=data_range, export=export)

        out = self.run_analysis()

        self.experiment.save_class()
        # need to move save_class() to here, because it can't be done in the experiment any more!

        return out

    def _autocorrelation_time(self):
        """
        Not needed in this analysis
        """
        raise NotApplicableToAnalysis

    def _get_rdf_data(self):
        """
        Fill the data_files list with filenames of the rdf tensor_values
        """
        database = AnalysisDatabase(name=os.path.join(self.experiment.database_path, "analysis_database"))
        self.data_files = database.get_tables("Radial_Distribution_Function")

    def _load_rdf_from_file(self):
        """
        Load the raw rdf tensor_values from a directory
        """
        database = AnalysisDatabase(name=os.path.join(self.experiment.database_path, "analysis_database"))
        data = database.load_pandas(self.file_to_study).to_numpy()
        self.radii = data[1:, 1]
        self.rdf = data[1:, 2]

    def _calculate_kb_integral(self):
        """
        calculate the Kirkwood-Buff integral
        """

        self.kb_integral = []  # empty the integration tensor_values

        for i in range(1, len(self.radii)):
            self.kb_integral.append(4*np.pi*(np.trapz((self.rdf[1:i] - 1)*(self.radii[1:i])**2, x=self.radii[1:i])))

    def run_post_generation_analysis(self):
        """
        Calculate the potential of mean-force and perform error analysis
        """

        self._get_rdf_data()  # fill the tensor_values array with tensor_values

        for data in self.data_files:
            self.file_to_study = data        # Set the file to study
            self.species_tuple = "_".join([data.split("_")[-1], data.split("_")[-2]])
            self.data_range = int(data.split("_")[-3])
            self._load_rdf_from_file()       # Load the rdf tensor_values for the set file
            self._calculate_kb_integral()    # Integrate the rdf and calculate the KB integral

            # Plot if necessary
            if self.plot:
                plt.plot(self.radii[1:], self.kb_integral, label=f"{self.species_tuple}")
                self._plot_data(title=f"{self.analysis_name}_{self.species_tuple}")

            if self.save:
                self._save_data(name=self._build_table_name(self.species_tuple),
                                data=self._build_pandas_dataframe(self.radii[1:], self.kb_integral))
            if self.export:
                self._export_data(name=self._build_table_name(self.species_tuple),
                                  data=self._build_pandas_dataframe(self.radii[1:], self.kb_integral))

    def _calculate_prefactor(self, species: Union[str, tuple] = None):
        """
        calculate the calculator pre-factor.

        Parameters
        ----------
        species : str
                Species property if required.
        Returns
        -------

        """
        raise NotImplementedError

    def _apply_operation(self, data, index):
        """
        Perform operation on an ensemble.

        Parameters
        ----------
        One tensor_values range of tensor_values to operate on.

        Returns
        -------

        """
        raise NotImplementedError

    def _apply_averaging_factor(self):
        """
        Apply an averaging factor to the tensor_values.
        Returns
        -------
        averaged copy of the tensor_values
        """
        raise NotImplementedError

    def _post_operation_processes(self, species: Union[str, tuple] = None):
        """
        call the post-op processes
        Returns
        -------

        """
        raise NotImplementedError

    def _update_output_signatures(self):
        """
        After having run _prepare managers, update the output signatures.

        Returns
        -------

        """
        raise NotImplementedError