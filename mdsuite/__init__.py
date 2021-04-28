"""
Python __init__ file
"""
import logging
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from .project import *  # Load the project class
from .experiment.experiment import Experiment
from .graph_modules import adjacency_matrix
from .utils.report_computer_characteristics import Report

# there is a good chance, that the root logger is already defined so we have to make some changes to it!
root = logging.getLogger()

# remove all existing loggers and overwrite them with ours
while root.hasHandlers():
    root.removeHandler(root.handlers[0])

root.setLevel(logging.DEBUG)  # <- seems to be the lowest possible loglevel so we set it to debug here!
# if we set it to info, we can not set the file oder stream to debug


# Logging to the stdout (Terminal, Jupyter, ...)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.INFO)  # <- stdout loglevel

formatter = logging.Formatter('%(asctime)s - %(name)s (%(levelname)s) - %(message)s')
stream_handler.setFormatter(formatter)
# attaching the stdout handler to the configured logging
root.addHandler(stream_handler)
