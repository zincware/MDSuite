Apple Silicon
-------------
Apple recently released it's new silicon for their latest lineup of products.
While it is great to see more high performing quality chips in the market, it
means there is work to be done ensuring all our favourite packages can
utilize this new technology. At the time of writing this installation of
most MDSuite dependencies is not too hard on a Mac M1, but we thought we would
add some documentation just to be helpful.

Getting the Source
^^^^^^^^^^^^^^^^^^
Unfortunately the Mac M1 install is currently only possible from source. To get started we need to get the source
code from the repository. Open a terminal and run the following:

.. code-block:: bash

   git clone https://github.com/zincware/MDSuite.git
   cd MDSuite

This should download the MDSuite code and change you into the main directly where you will see the file structure.

TensorFlow Install
^^^^^^^^^^^^^^^^^^
**The following is taken from issue #153 from the apple/tensorflow_macos repository found**
`here <https://github.com/apple/tensorflow_macos/issues/153>`_

In order to satisfy all of the dependencies we will need to use MiniForge to create a virtual environment.

1.) Install MiniForge

    Go `here <https://github.com/conda-forge/miniforge#miniforge3>`_ and install MiniForge for your machine. Once it has
    been installed you should run some checks to make sure your commands are pointing to the correct executables.

    .. code-block:: bash

       which python  # should point to ~/miniforge/bin/python.
       which pip  # should point to a miniforge directory.

    If this is not the case, you will need to look into adding these paths to your ~/.zshrc or ~/.bashrc or whichever
    rc file you are using.

2.) Download :download:`environment.yml <../_static/environment.yml>`

3.) Go to your terminal and run the following commands from the directory in which you have the environment.yml file:

    .. code-block:: bash

       conda env create --file=environment.yml --name=tf_macos
       conda activate tf_macos
       pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_addons_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha2/tensorflow_macos-0.1a2-cp38-cp38-macosx_11_0_arm64.whl

4.) Go and test your installation by running:

    .. code-block:: bash

       python
       >>> import tensorflow as tf

    If this does not fail, the installation worked!

Dependencies Installation
^^^^^^^^^^^^^^^^^^^^^^^^^
This is where the installation can get a little bit difficult. Several MDSuite packages will require conda installation
rather than pip due to their use of a FORTRAN compiler. This can be achieved by running the following:

.. code-block:: bash

   conda install numpy
   conda install scipy
   conda install pandas
   conda install matplotlib
   pip install h5py

This should take care of the difficult installations. There is still one more step to the installation just to make sure
nothing goes wrong when we run pip install on the MDSuite directory.

Updating requirements.txt
^^^^^^^^^^^^^^^^^^^^^^^^^
We need to remove some of the dependencies from the requirements.txt file in order to make sure pip doesn't try to
uninstall and reinstall packages that require conda. Open the requirements.txt file in the main directory of MDSuite
and remove the following packages from it:

* tensorflow
* h5py
* numpy
* scipy
* pandas
* matplotlib

Once this is done, run the following to complete the install:

.. code-block:: bash

   pip install .

This will install MDSuite to the directory path which makes it a little easier to update the code.

Concluding Remarks
^^^^^^^^^^^^^^^^^^
With that, you should have installed MDSuite on the new Mac silicon. Enjoy playing around and checking out the
performance of this new technology. It is expected in the future that the Apple branch of TensorFlow will be merged
with the main package. Ideally this takes place in parallel with support for the other packages required for MDSuite.
We will keep updating these docs to reflect the current state of the Mac M1 install so check back in from time to time
to see if it has become a little easier.
