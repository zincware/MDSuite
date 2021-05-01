"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the MDSuite Project.
"""

"""
Python __init__ file
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # Load tensorflow early to avoid later delays
from .project import *  # Load the project class
from .experiment.experiment import Experiment
from .graph_modules import adjacency_matrix
from .utils.report_computer_characteristics import Report
