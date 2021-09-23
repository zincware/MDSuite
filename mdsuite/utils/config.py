"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Configuration for MDSuite
"""
from dataclasses import dataclass


@dataclass
class Config:
    """Collection of MDSuite configurations"""
    jupyter: bool = False
    GPU: bool = False


config = Config()

try:
    config.jupyter = get_ipython().__class__.__name__ == "ZMQInteractiveShell"
except NameError:
    pass
