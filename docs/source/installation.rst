Installation
------------

MDSuite may be installed either directly from pip or from source.

Pip Install
===========
MDSuite is hosted on the PyPi repository and there can be installed by the following:

.. code-block:: bash

   pip install mdsuite

Alternatively, you may wish to install the main branch directly from source.

From Source
===========

From the directory in which you wish to store MDSuite, run

.. code-block:: bash

        git clone https://github.com/SamTov/MDSuite.git

Once the directory has finished cloning, change into the MDSuite directory,
and run

.. code-block:: bash

   pip3 install .

Or if you want to install as developer:

.. code-block:: bash

   pip3 install -e .

which will install the mdsuite package as a python library. Once this has
been done, you can simply call mdsuite from any of your python projects with

.. code-block:: python

   import mdsuite

or with an alias you prefer. We usually run with

.. code-block:: python

   import mdsuite as mds

Once you have installed the package, go and check out the tutorials directory
to see how you can start using it to analyze your own simulations.
