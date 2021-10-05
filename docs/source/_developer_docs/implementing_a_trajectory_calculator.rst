Implementing a Trajectory Calculator
------------------------------------
The more involved calculator is certainly the trajectory calculator.
This is due to the interfacing with the MDSuite database as well as considering memory
safety and data handling.
This has all been wrapped up into the parent class but certain steps must be taken to
make the most of the infrastructure.


Imports
=======
The first step to many python programs are the import statements.
In the case of an MDSuite calculator the important ones are as follows:

.. code-block:: python

   import logging
   from mdsuite.calculators import call
   from dataclasses import dataclass
   from mdsuite.database import simulation_properties
   from mdsuite.calculators import TrajectoryCalculator

   log = logging.getLogger(__name__)

Let's go through them one by one.

* logging is the library you will want to use to log information. Typically, when you
  want to output some information you might use a print statement. However, by using
  for example :code:`log.info` or :code:`log.warning` you can better utilize Python
  software frameworks.
* call is a decorator method we have implemented which helps to take care of some
  commonly called methods between calculators such as saving data into the SQL database.
* dataclass will be used to define a dataclass for the properties that are saved into
  the MDSuite project database.
* simulation_properties is a dataclass of all the different properties that can be
  stored in an MDSuite simulation database. If your calculator is using data that you
  have specially made and added, you will need to include the name of the HDF5 database
  group to this dataclass. This will all be covered in the
  'Implementing a Transformation' and 'Implementing a File Reader' tutorials.
* TrajectoryCalculator is the parent class for all of our trajectory calculators.
  It contains within it all the necessary methods for memory and data management and
  also inherits from the main Calculator parent class.

Dataclass
=========
One of the great things about MDSuite is that it stores the outcome of all computations
in an SQL database along with the meta-data of the calculation.
In order for this to happen, the calculator needs to know what should be saved.
We do this using a dataclass for each calculator, the contents of which will end up
in the project database.

.. code-block:: python

   @dataclass
   class Args:
       data_range: int
       correlation_time: int
       atom_selection: np.s_
       tau_values: np.s_
       species: list

Here we have shown the dataclass from our Einstein diffusion coefficients.
In this calculator, we want to store the data_range over which the function was measured,
the correlation time used, the atoms selected for the computation, the time steps used,
and the species studied in the analysis.
Consider now the dataclass for the radial distribution function:

.. code-block:: python

   @dataclass
   class Args:
       number_of_bins: int
       number_of_configurations: int
       atom_selection: np.s_
       cutoff: float
       start: int
       stop: int
       species: list

In this case, any of the information related with data ranges or correlation times is
removed in favour of knowing how many configurations and bins were used in the analysis.
This isn't the end of the story for the SQL database, there is one more part that we
will discuss shortly.

Declaration and __init__
========================
We can now define the class.
In reality, the :code:`__init__` method of MDSuite calculators is not overly useful as we rely
mostly on the :code:`__call__` method.
However, as best practice dictates, we must still declare our class variables in the
:code:`__init__` and for your own benefit, should document them there.

.. code-block:: python

   class NewTrajectoryCalculator(TrajectoryCalculator):
       """
       Class for the Einstein diffusion coefficient implementation

       Notes
       -----

       Attributes
       ----------

       See Also
       --------

       Examples
       --------
       """

       def __init__(self, **kwargs):
           """

           Parameters
           ----------

           """

           super().__init__(**kwargs)
           self.scale_function = {"linear": {"scale_factor": 150}}
           self.loaded_property = simulation_properties.unwrapped_positions
           self.x_label = r"$ x label $"
           self.y_label = r"$ y label $"
           self.result_keys = ["diffusion_coefficient", "uncertainty"]
           self.result_series_keys = ["time", "msd"]
           self.analysis_name = "Name of Analysis"
           self.data_array = None
           self.tau_values = None
           self._dtype = tf.float64

           log.info("starting ... Computation")

This barebones example was taken from the einstein diffusion coefficient calculator
where we have stripped out a lot of doc-strings and removed some string in place of a
more descriptive option.
So now let's go through these attributes one by one and discuss briefly what they are
doing and if they are necessary.
In the following list, all of the highlighted attributes must be defined.

* :code:`scale_function`: This is how the calculator scales with the data input and is
  essential for the memory management.
  A list of scale functions can be found in mdsuite.utils.scale_functions
* :code:`loaded_property`: This is the data that will be loaded from the database.
  It references a tuple of the form ("Positions", (None, None, 3)).
  The first string is the name of the HDF5 database group that will be loaded and the
  second tuple is shape of the data.
  Note that None contained here is correct.
  You should NOT put an actual value here.
  The only important thing is that the number of elements is correct.
  Given that each atom has a position for all time-steps, there are three elements.
  If something like ionic_current is being studied, which has only one value for each
  configuration, the parameter will look like ("Ionic_Current", (None, 3)).
* :code:`x_label`: Name used on the x axis of the plot.
* :code:`y_label`: Name used on the y axis of the plot.
* :code:`result_keys`: This is what the single value data will be stored as. In the case
  of an einstein diffusion coefficient, both the actual diffusion coefficient and the
  msd will be plotted. In this case, we want to store the diffusion coefficient and the
  uncertainty and so that is how we label the data.
* :code:`results_series_keys`: These keys are the names of the series data. Following
  the previous example, they are called time and msd.
* :code:`analysis_name`: This is the name of the analysis so that it can be labelled
  correctly. Could be radial distribution function or einstein diffusion coefficient.
* :code:`_dtype`: Type required in the analysis e.g. :code:`tf.float64`.
* data_array: For most analysis you will loop over batches or ensembles or both.
  In this case, it is easiest at each iteration to update a class attribute than it
  is to handle returns.
  Therefore, I have a data array here of, in the diffusion case, my msd.
* tau_values: When you run an msd over 500 time steps sometimes you will want to use
  only every n steps.
  Tau_values is the parameter you wil set if you do not want to use every time step
  between 0 and your data range.

The __call__ method
===================
Python has a nice call method which allows a class to be called as ClassName().
MDSuite makes use of this to allow for things like autocomplete as well as streamlined
execution through a parent class.
The call method in an MDSuite calculator takes on the form of a standard :code:`__init`
and is where user inputs are processed.

.. code-block:: python

   @call
   def __call__(
       self,
       plot: bool = True,
       species: list = None,
       data_range: int = 100,
       correlation_time: int = 1,
       atom_selection: np.s_ = np.s_[:],
       molecules: bool = False,
       tau_values: Union[int, List, Any] = np.s_[:],
       gpu: bool = False,
   ):
       """

       Parameters
       ----------

       Returns
       -------
       """
       if species is None:
           if molecules:
               species = list(self.experiment.molecules)
           else:
               species = list(self.experiment.species)
       # set args that will affect the computation result
       self.args = Args(
           data_range=data_range,
           correlation_time=correlation_time,
           atom_selection=atom_selection,
           tau_values=tau_values,
           molecules=molecules,
           species=species,
       )
       self.gpu = gpu
       self.plot = plot
       self.system_property = False
       self.time = self._handle_tau_values()
       self.msd_array = np.zeros(self.data_resolution)

Again we have removed all of the doc-strings for clarity.
In the call method, not only do we populate some of the declared methods from the
:code:`__init__` but we also define some new ones and populate the database attributes
that we defined earlier.
In the call itself the user arguments should be passed.
