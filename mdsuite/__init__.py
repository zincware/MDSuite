"""
MDSuite: A Zincwarecode package.

License
-------
This program and the accompanying materials are made available under the terms
of the Eclipse Public License v2.0 which accompanies this distribution, and is
available at https://www.eclipse.org/legal/epl-v20.html

SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincwarecode Project.

Contact Information
-------------------
email: zincwarecode@gmail.com
github: https://github.com/zincware
web: https://zincwarecode.com/

Citation
--------
If you use this module please cite us with:

Summary
-------
"""
import logging
import sys
from .project import Project
from .experiment import Experiment
from .graph_modules import adjacency_matrix
from .utils.report_computer_characteristics import Report
from .utils import config

__all__ = ['Project', 'Experiment', 'adjacency_matrix', 'Report', 'config']

logger = logging.getLogger("mdsuite")
logger.setLevel(logging.INFO)

# Formatter for advanced logging
formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')

channel = logging.StreamHandler(sys.stdout)
channel.setLevel(logging.DEBUG)
channel.setFormatter(formatter)

logger.addHandler(channel)
