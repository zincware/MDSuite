# Introduction
MDSuite is a software designed specifically for the molecular dynamics community to enable fast, reliable, and simple 
calculations from simulation data. 

**NOTE:** This README is designed for the release. Anything PyPi related is not yet functioning as the code is in
development.

## Installation
There are several way to install MDSuite depending on what you would like from it. One can simply installing using 
PyPi as **Not Currently available**
```bash
$ pip install mdsuite
```
If you would like to install it from source then you can clone the repository by running
```bash
$ git clone https://github.com/SamTov/MDSuite.git
```
Once you have cloned the repository, depending on whether you prefer conda or straight pip, you should follow the 
instructions below.

### Installation with pip
```bash
$ cd MDSuite
$ pip install .
``` 
### Installation with conda
```bash
$ cd MDSuite
$ conda create -n MDSuite python=3.8
$ conda activate MDSuite
$ pip install . 
```

### Important installation notes
NOTE: Tensorflow currently requires h5py < 3.0 but MDSuite requires h5py 3.0 or later because of new memory management 
features. Therefore, it is currently necessary to run ``pip install h5py --upgrade --no-dependencies`` after the 
installation. See https://github.com/tensorflow/tensorflow/issues/47303

NOTE: to install tensorflow with GPU support, use CUDA 10.1 and cuDNN 7.6.5. 

This has been written into the setup.py and so should NOT require any additional work 

## Documentation
There is a live version of the documentation hosted [here](https://mdsuite.readthedocs.io/en/latest/).
If you would prefer to have a local copy, it can be built using sphinx by following the instructions below.
```bash
$ cd MDSuite/docs
$ make html
$ cd build/html
$ firefox index.html
```
