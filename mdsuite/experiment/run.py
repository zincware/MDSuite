"""
This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0

Copyright Contributors to the Zincware Project.

Description: Collection of calculators / transformations for exp.run
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Type, Union, Any

from mdsuite.calculators import (
    AngularDistributionFunction,
    CoordinationNumbers,
    EinsteinDiffusionCoefficients,
    EinsteinDistinctDiffusionCoefficients,
    EinsteinHelfandIonicConductivity,
    EinsteinHelfandThermalConductivity,
    EinsteinHelfandThermalKinaci,
    GreenKuboDiffusionCoefficients,
    GreenKuboDistinctDiffusionCoefficients,
    GreenKuboIonicConductivity,
    GreenKuboThermalConductivity,
    GreenKuboViscosity,
    GreenKuboViscosityFlux,
    KirkwoodBuffIntegral,
    NernstEinsteinIonicConductivity,
    PotentialOfMeanForce,
    RadialDistributionFunction,
    SpatialDistributionFunction,
    StructureFactor,
)

from mdsuite.transformations import CoordinateWrapper, Transformations, CoordinateUnwrapper

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment


class RunComputation:
    """Collection of all calculators that can be used by an experiment"""

    def __init__(
        self, experiment: Experiment = None, experiments: List[Experiment] = None
    ):
        """Collection of all calculators

        Parameters
        ----------
        experiment: Experiment
            Experiment to run the computations for
        experiments: List[Experiment]
            A list of experiments passed by running the computation from the project
            class
        """
        self.experiment = experiment
        self.experiments = experiments

        self.kwargs = {"experiment": experiment, "experiments": experiments}

    def exp_wrapper(self, func):
        # preparation for https://github.com/zincware/MDSuite/issues/404
        # currently this does nothing
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_instance = func(*args, **kwargs)
            # self.experiment._run(func_instance)
            return func_instance

        return wrapper

    def transformation_wrapper(self, func: Union[Type[Transformations], Any]):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if self.experiments is None:
                self.experiments = [self.experiment]
            for experiment in self.experiments:
                func_instance = func(*args, **kwargs)
                experiment.cls_transformation_run(func_instance)
            return None
        return wrapper


    @property
    def CoordinateWrapper(self) -> CoordinateWrapper:
        return self.transformation_wrapper(CoordinateWrapper)

    @property
    def CoordinateUnwrapper(self) -> CoordinateUnwrapper:
        return self.transformation_wrapper(CoordinateUnwrapper)

    @property
    def AngularDistributionFunction(self) -> AngularDistributionFunction:
        """Calculator Property"""
        return self.exp_wrapper(AngularDistributionFunction)(**self.kwargs)

    @property
    def CoordinationNumbers(self) -> CoordinationNumbers:
        """Calculator Property"""
        return self.exp_wrapper(CoordinationNumbers)(**self.kwargs)

    @property
    def EinsteinDiffusionCoefficients(self) -> EinsteinDiffusionCoefficients:
        """Calculator Property"""
        return self.exp_wrapper(EinsteinDiffusionCoefficients)(**self.kwargs)

    @property
    def EinsteinDistinctDiffusionCoefficients(
        self,
    ) -> EinsteinDistinctDiffusionCoefficients:
        """Calculator Property"""
        return self.exp_wrapper(EinsteinDistinctDiffusionCoefficients)(**self.kwargs)

    @property
    def EinsteinHelfandIonicConductivity(self) -> EinsteinHelfandIonicConductivity:
        """Calculator Property"""
        return self.exp_wrapper(EinsteinHelfandIonicConductivity)(**self.kwargs)

    @property
    def EinsteinHelfandThermalKinaci(self) -> EinsteinHelfandThermalKinaci:
        """Calculator Property"""
        return self.exp_wrapper(EinsteinHelfandThermalKinaci)(**self.kwargs)

    @property
    def GreenKuboViscosityFlux(self) -> GreenKuboViscosityFlux:
        """Calculator Property"""
        return self.exp_wrapper(GreenKuboViscosityFlux)(**self.kwargs)

    @property
    def GreenKuboDistinctDiffusionCoefficients(
        self,
    ) -> GreenKuboDistinctDiffusionCoefficients:
        """Calculator Property"""
        return self.exp_wrapper(GreenKuboDistinctDiffusionCoefficients)(**self.kwargs)

    @property
    def GreenKuboIonicConductivity(self) -> GreenKuboIonicConductivity:
        """Calculator Property"""
        return self.exp_wrapper(GreenKuboIonicConductivity)(**self.kwargs)

    @property
    def GreenKuboDiffusionCoefficients(self) -> GreenKuboDiffusionCoefficients:
        """Calculator Property"""
        return self.exp_wrapper(GreenKuboDiffusionCoefficients)(**self.kwargs)

    @property
    def GreenKuboThermalConductivity(self) -> GreenKuboThermalConductivity:
        """Calculator Property"""
        return self.exp_wrapper(GreenKuboThermalConductivity)(**self.kwargs)

    @property
    def GreenKuboViscosity(self) -> GreenKuboViscosity:
        """Calculator Property"""
        return self.exp_wrapper(GreenKuboViscosity)(**self.kwargs)

    @property
    def KirkwoodBuffIntegral(self) -> KirkwoodBuffIntegral:
        """Calculator Property"""
        return self.exp_wrapper(KirkwoodBuffIntegral)(**self.kwargs)

    @property
    def NernstEinsteinIonicConductivity(self) -> NernstEinsteinIonicConductivity:
        """Calculator Property"""
        return self.exp_wrapper(NernstEinsteinIonicConductivity)(**self.kwargs)

    @property
    def PotentialOfMeanForce(self) -> PotentialOfMeanForce:
        """Calculator Property"""
        return self.exp_wrapper(PotentialOfMeanForce)(**self.kwargs)

    @property
    def RadialDistributionFunction(self) -> RadialDistributionFunction:
        """Calculator Property"""
        return self.exp_wrapper(RadialDistributionFunction)(**self.kwargs)

    @property
    def StructureFactor(self) -> StructureFactor:
        """Calculator Property"""
        return self.exp_wrapper(StructureFactor)(**self.kwargs)

    @property
    def EinsteinHelfandThermalConductivity(self) -> EinsteinHelfandThermalConductivity:
        """Calculator Property"""
        return self.exp_wrapper(EinsteinHelfandThermalConductivity)(**self.kwargs)

    @property
    def SpatialDistributionFunction(self) -> SpatialDistributionFunction:
        """Calculator Property"""
        return self.exp_wrapper(SpatialDistributionFunction)(**self.kwargs)
