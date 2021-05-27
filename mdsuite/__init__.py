"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.

Python __init__ file
"""
import logging
import os
import sys
from .project import Project
from .experiment.experiment import Experiment
from .graph_modules import adjacency_matrix
from .utils.report_computer_characteristics import Report

__all__ = ['Project', 'Experiment', 'adjacency_matrix', 'Report']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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

formatter = logging.Formatter('%(asctime)s (%(levelname)s) - %(message)s')
stream_handler.setFormatter(formatter)
# attaching the stdout handler to the configured logging
root.addHandler(stream_handler)
