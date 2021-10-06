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
The first real distinction between system and species-wise calculators is present here.
Because these calculators run for a specific species, you have to handle if the user
has not added the data themselves.
In this case, we take all species or molecules in the system as the default.
Considering the :code:`__call__` argument from the ionic conductivity below we can
see the difference.

.. code-block:: python

   @call
   def __call__(
       self,
       plot=True,
       data_range=500,
       correlation_time=1,
       tau_values: np.s_ = np.s_[:],
       gpu: bool = False,
   ):
       """
       Python constructor

       Parameters
       ----------
       plot : bool
               if true, plot the tensor_values
       data_range :
               Number of configurations to use in each ensemble
       correlation_time : int
               Correlation time to use in the analysis.
       gpu : bool
               If true, reduce memory usage to the maximum GPU capability.
       """
       # set args that will affect the computation result
       self.args = Args(
           data_range=data_range,
           correlation_time=correlation_time,
           tau_values=tau_values,
           atom_selection=np.s_[:],
       )

       self.gpu = gpu
       self.plot = plot
       self.time = self._handle_tau_values()
       self.msd_array = np.zeros(self.data_resolution)

The next important part is the updating of the :code:`self.args` data class.
Remember that whatever is added here will be recorded in the SQL database and used for
querying later.
Beyond this, when you need to call one of these properties during the calculator, you
must call it through the data class by :code:`self.args.data-range`.
The remaining attributes are those that are either used only within the class such as
:code:`self.msd_array` and :code:`self.time` or are used in the parent class such as
:code:`self.gpu` and :code:`self.plot`.
These last 4 lines are quite important and so we will discuss them in more detail.

* :code:`self.gpu`: At the moment the :code:`self.gpu` command only reduces the max
  memory allowed to that of the biggest GPU.
* :code:`self.plot`: If this is set to false, no plots of the analysis will be
  generated.
* :code:`self.time`: This attribute is set by calling the
  :code:`self._handle_tau_values` method which will allow for the use of custom time
  steps within a data range.
  This then correlates directly to the :code:`self.msd_array` which must be instantiated
  with the :code:`self.data_resolution` attribute.
  This attribute is an integer describing how many points are loaded between 0 and the
  data_range.

The run_calculator Method
=========================
Finally we can discuss the running of the calculator including the loading of data and
the implementation of the operation.

In this section we will break down three different examples.
The first two will be very similar whereas the third will be an alternative to
implementing calculators.

Species calculators
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   def run_calculator(self):
       """
       Run analysis.

       Returns
       -------

       """
       self.check_input()
       # Loop over species
       for species in self.args.species:
           dict_ref = str.encode("/".join([species, self.loaded_property[0]]))
           self.calculate_prefactor(species)

           batch_ds = self.get_batch_dataset([species])
           for batch in tqdm(
               batch_ds,
               ncols=70,
               desc=species,
               total=self.n_batches,
               disable=self.memory_manager.minibatch,
           ):
               ensemble_ds = self.get_ensemble_dataset(batch, species, split=True)

               for ensemble in ensemble_ds:
                   self.ensemble_operation(ensemble[dict_ref])

           # Scale, save, and plot the data.
           self.postprocessing(species)

The first line in this calculator runs a check on the user input.
This is optional but advisable if inputs that can kill the analysis are possible.
The second part, and the main difference between system-wise and species-wise
calculators, is the loop over the chosen species.
What follows can be broken into four components.

1. Get a batch dataset. This is the first step in memory management. A batch is N
   configurations that can fit into memory keeping in mind what kind of inflation the
   operation will cause.
2. Get an ensemble dataset for that batch. An ensemble is a subset of the N loaded
   configuration over which you actually perform a computation. Consider the MSD on
   a data_range of 500. It may be faster and possible to load 1000 configurations in and
   loop over ensembles of 500 configurations sliding along in a window in steps of
   correlation time.
3. Perform an operation on each ensemble. This can include computing the msd or
   performing auto-correlation.
4. Run some post-processing on the analysis including plotting.

The most important point here is what do you have to pass to the batch data set method
and what you get out.
The argument it takes is a list of species from which data should be loaded.
The dataset will know what data is to be loaded based on the :code:`loaded_property`
attribute in the class.
What you get back is a TensorFlow Dataset.
This is a generator object which is also capable of pre-fetching data between loops for
maximum performance.
When you loop over ths dataset you will get a dict object back with byte-string encoded
keys.
This is due to some back-end TensorFlow processes and is the reason we defined a
dict_ref variable at the start of the loop.
For example, this is what would happen if you wished to load position data for
:code:`Na` and :code:`Cl`:

.. code-block:: python

   ds = self.get_batch_dataset(["Na", "Cl"])
   for batch in ds:
       print(ds)

   >>> ds = {b'Na/Positions': tf.Tensor(shape=(n_atoms, n_confs, 3)),
             b'Cl/Positions': tf.Tensor(shape=(n_atoms, n_confs, 3)),
             b'data_size': tf.int32}

Typically the data_size is set to None but will be returned with the loading.
It is important to remember this if you plan on looping over this dict.

System Calculators
^^^^^^^^^^^^^^^^^^
Now we will look at the run method of a system calculator.
You will notice that the only difference ins the species loop as well as what is passed
to the batch generator.

.. code-block:: python

   def run_calculator(self):
       """

       Run analysis.

       Returns
       -------

       """
       self.check_input()
       # Compute the pre-factor early.
       self._calculate_prefactor()

       dict_ref = str.encode(
           "/".join([self.loaded_property[0], self.loaded_property[0]])
       )

       batch_ds = self.get_batch_dataset([self.loaded_property[0]])

       for batch in tqdm(
           batch_ds,
           ncols=70,
           total=self.n_batches,
           disable=self.memory_manager.minibatch,
       ):
           ensemble_ds = self.get_ensemble_dataset(batch, self.loaded_property[0])

           for ensemble in ensemble_ds:
               self.ensemble_operation(ensemble[dict_ref])

       # Scale, save, and plot the data.
       self._apply_averaging_factor()
       self._post_operation_processes()

So, as mentioned, there is no species loop.
Beyond this,the loaded property is what is passed to the batch generator.
This is because what is stored in the HDF5 database is under a group with the same name.
For example, ionic current is stored as :code:`db["Ionic_Current"/"Ionic_Current"]`.

None-ensemble calculators.
^^^^^^^^^^^^^^^^^^^^^^^^^^
In cases where one wants to load several species at the same time the ensemble ds will
likely not suffice.
All of the MDSuite structural calculators have this problem and therefore look slightly
different.
Let's look at the RDF to see what is going on.

.. code-block:: python

   def run_calculator(self):
       """
       Run the analysis.

       Returns
       -------

       """
       self.check_input()

       dict_keys, split_arr, batch_tqm = self.prepare_computation()

       # Get the batch dataset
       batch_ds = self.get_batch_dataset(
           subject_list=self.args.species, loop_array=split_arr, correct=True
       )

       # Loop over the batches.
       for idx, batch in tqdm(enumerate(batch_ds), ncols=70, disable=batch_tqm):

           # Reformat the data.
           log.debug("Reformatting data.")
           positions_tensor = self._format_data(batch=batch, keys=dict_keys)

           # Create a new dataset to loop over.
           log.debug("Creating dataset.")
           per_atoms_ds = tf.data.Dataset.from_tensor_slices(positions_tensor)
           n_atoms = tf.shape(positions_tensor)[0]

           # Start the computation.
           log.debug("Beginning calculation.")
           minibatch_start = tf.constant(0)
           stop = tf.constant(0)
           rdf = {
               name: tf.zeros(self.args.number_of_bins, dtype=tf.int32)
               for name in self.key_list
           }

           for atoms in tqdm(
               per_atoms_ds.batch(self.minibatch).prefetch(tf.data.AUTOTUNE),
               ncols=70,
               disable=not batch_tqm,
               desc=f"Running mini batch loop {idx + 1} / {self.n_batches}",
           ):
               # Compute the minibatch update
               minibatch_rdf, minibatch_start, stop = self.run_minibatch_loop(
                   atoms, stop, n_atoms, minibatch_start, positions_tensor
               )

               # Update the rdf.
               start_time = timer()
               rdf = self.combine_dictionaries(rdf, minibatch_rdf)
               log.debug(f"Updating dictionaries took {timer() - start_time} s")

           # Update the class before the next batch.
           for key in self.rdf:
               self.rdf[key] += rdf[key]

       self._calculate_radial_distribution_functions()

In these cases after the batch is generated some alternative mini-batching is applied
to ensure memory safety.
In principle however, once the batch is being looped over you can apply whatever
operations you want.

Post-processing
---------------
