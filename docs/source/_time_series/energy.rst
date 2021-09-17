Plot Energy over Time
=====================

This can be used to plot the energy of a trajectory over time.
Assume you have an experiment `exp`:

.. code-block:: python

    exp.analyse_time_series.Energies(species=["K", "Cl"], rolling_window=5)


This will plot the energy over time.