import setuptools
from setuptools import find_packages
from distutils.core import setup, Extension
from Cython.Build import cythonize

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
                      'h5py',
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'tqdm',
                      'psutil',
                      'tensorflow>=2.1',
                      'numpy'])
