"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Plot the time series energies of the trajectory
"""
from __future__ import annotations
from .base import TimeSeries

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mdsuite import Experiment


class Energies(TimeSeries):
    def __init__(self, experiment: Experiment):
        super().__init__(experiment)
        self.loaded_property = "PE"
        self.fig_labels = {"x": r"Timestep $t$", "y": r"Energy $E$"}
