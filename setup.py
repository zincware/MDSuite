import setuptools

from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess
import sys


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        develop.run(self)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py", "--upgrade", "--no-dependencies"])


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        install.run(self)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "h5py", "--upgrade", "--no-dependencies"])


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MDSuite",
    version="0.0.1",
    author="Samuel Tovey",
    author_email="tovey.samuel@gmail.com",
    description="A postprocessing tool for molecular dynamics simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SamTov/MDSuite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_data={'': ['form_fac_coeffs.csv']},
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['seaborn',
                      'tensorflow',
                      'h5py',
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'tqdm',
                      'psutil',
                      'numpy',
                      'gputil',
                      'diagrams',
                      'pubchempy',
                      'PyYAML',
                      'sphinxcontrib.bibtex',
                      'pybtex',
                      'dask',
                      'nbsphinx',
                      'sphinx_rtd_theme',
                      'ipython',
                      'numpydoc',
                      'sphinx-copybutton'],
    cmdclass={'install': PostInstallCommand, 'develop': PostDevelopCommand}
    # force install of the newest h5py after the dependencies are installed
    # See https://github.com/tensorflow/tensorflow/issues/47303 for further information
    # TODO remove if tensorflow supports h5py > 3
)
