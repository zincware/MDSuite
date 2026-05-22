"""MDSuite run module.

This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0.

Copyright Contributors to the Zincware Project.

Description: Dispatch object exposed by ``experiment.run`` / ``project.run``.

This module also implements the back-compat shim that lets the legacy
``experiment.run.<Calculator>(**kwargs)`` API keep working after the
calculator refactor:

    # New API (standalone calculator → applied to experiment(s))
    calc = mds.GreenKuboDiffusionCoefficients(data_range=500, plot=True)
    result = experiment.run(calc)
    results = project.run(calc, experiments=[exp1, exp2])

    # Legacy API (still supported via shim)
    experiment.run.GreenKuboDiffusionCoefficients(data_range=500, plot=True)
"""
from __future__ import annotations

import concurrent.futures
import copy
import functools
from typing import TYPE_CHECKING, Any, List, Type, Union

import mdsuite.database.scheme as db
from mdsuite.calculators import (
    AngularDistributionFunction,  # SpatialDistributionFunction,
)
from mdsuite.calculators import (
    Calculator,
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
    PotentialOfMeanForce,
    RadialDistributionFunction,
    StructureFactor,
)
from mdsuite.transformations import (
    CoordinateUnwrapper,
    CoordinateWrapper,
    IntegratedHeatCurrent,
    IonicCurrent,
    KinaciIntegratedHeatCurrent,
    MolecularMap,
    MomentumFlux,
    ScaleCoordinates,
    ThermalFlux,
    Transformations,
    TranslationalDipoleMoment,
    UnwrapViaIndices,
    VelocityFromPositions,
)

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment


class RunComputation:
    """Dispatch object returned by ``experiment.run`` / ``project.run``.

    Supports two call styles:

    1. **New (preferred)** — pass a standalone calculator instance::

           calc = mds.GreenKuboDiffusionCoefficients(data_range=500, plot=True)
           experiment.run(calc)        # → db.Computation
           project.run(calc)           # → {exp_name: db.Computation, ...}

    2. **Legacy** — call the property with kwargs, the shim constructs and
       runs the calculator for you::

           experiment.run.GreenKuboDiffusionCoefficients(data_range=500, plot=True)
    """

    def __init__(
        self, experiment: Experiment = None, experiments: List[Experiment] = None
    ):
        """
        Parameters
        ----------
        experiment: Experiment
            Single experiment to run computations against (used by
            ``experiment.run``).
        experiments: List[Experiment]
            Experiments to run computations against (used by ``project.run``).
        """
        self.experiment = experiment
        self.experiments = experiments

    def _target_experiments(self) -> List[Experiment]:
        """Return the experiments this dispatcher applies calculators to."""
        if self.experiments is not None:
            return list(self.experiments)
        if self.experiment is not None:
            return [self.experiment]
        return []

    def __call__(
        self,
        calculator: Calculator,
        parallel: bool = False,
        max_workers: int = None,
    ) -> Union[db.Computation, dict]:
        """Apply a standalone calculator instance to the target experiment(s).

        Parameters
        ----------
        calculator : Calculator
            A configured calculator instance.
        parallel : bool, default False
            Run experiments concurrently in a thread pool. Each worker
            operates on its own ``copy.deepcopy`` of the calculator, so
            instance state (``self.experiment``, ``self.args``,
            ``self.jacf``, ...) is isolated. Heavy compute releases the
            GIL through JAX / TF / NumPy, so threads — not processes —
            are the right primitive: they avoid the
            ``Experiment``-with-SQLAlchemy-session pickling problem and
            still scale across cores for CPU-bound JAX kernels. Has no
            effect when there is only one target experiment.
        max_workers : int, optional
            Override the default thread-pool size (the number of target
            experiments).

        Returns
        -------
        db.Computation or dict[str, db.Computation]
            A single computation result when called from
            ``experiment.run(calc)``; a ``{experiment_name: result}`` dict
            when called from ``project.run(calc)``.
        """
        targets = self._target_experiments()

        if parallel and len(targets) > 1:
            results = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers or len(targets)
            ) as pool:
                future_to_name = {
                    pool.submit(copy.deepcopy(calculator).run, exp): exp.name
                    for exp in targets
                }
                for future in concurrent.futures.as_completed(future_to_name):
                    results[future_to_name[future]] = future.result()
        else:
            results = {exp.name: calculator.run(exp) for exp in targets}

        if self.experiment is not None and self.experiments is None:
            # Single-experiment dispatch (experiment.run(calc))
            return results[self.experiment.name]
        return results

    def _calculator_shim(self, cls: Type[Calculator]):
        """Build a legacy-API shim for a calculator class.

        The shim accepts the same kwargs the calculator now takes in its
        ``__init__``, constructs the calculator, and runs it against the
        target experiment(s).
        """

        @functools.wraps(cls.__init__)
        def shim(**kwargs):
            calc = cls(**kwargs)
            return self(calc)

        return shim

    def transformation_wrapper(self, func: Union[Type[Transformations], Any]):
        """Run the transformation for every selected experiment."""

        @functools.wraps(func.run_transformation)
        def wrapper(*args, **kwargs):
            for experiment in self._target_experiments():
                func_instance = func()
                experiment.cls_transformation_run(func_instance, *args, **kwargs)

        return wrapper

    #######################
    ### Transformations ###
    #######################

    @property
    def CoordinateWrapper(self) -> Type[CoordinateWrapper]:
        return self.transformation_wrapper(CoordinateWrapper)

    @property
    def CoordinateUnwrapper(self) -> Type[CoordinateUnwrapper]:
        return self.transformation_wrapper(CoordinateUnwrapper)

    @property
    def IntegratedHeatCurrent(self) -> Type[IntegratedHeatCurrent]:
        return self.transformation_wrapper(IntegratedHeatCurrent)

    @property
    def IonicCurrent(self) -> Type[IonicCurrent]:
        return self.transformation_wrapper(IonicCurrent)

    @property
    def KinaciIntegratedHeatCurrent(self) -> Type[KinaciIntegratedHeatCurrent]:
        return self.transformation_wrapper(KinaciIntegratedHeatCurrent)

    @property
    def MolecularMap(self) -> Type[MolecularMap]:
        return self.transformation_wrapper(MolecularMap)

    @property
    def MomentumFlux(self) -> Type[MomentumFlux]:
        return self.transformation_wrapper(MomentumFlux)

    @property
    def ScaleCoordinates(self) -> Type[ScaleCoordinates]:
        return self.transformation_wrapper(ScaleCoordinates)

    @property
    def ThermalFlux(self) -> Type[ThermalFlux]:
        return self.transformation_wrapper(ThermalFlux)

    @property
    def TranslationalDipoleMoment(self) -> Type[TranslationalDipoleMoment]:
        return self.transformation_wrapper(TranslationalDipoleMoment)

    @property
    def UnwrapViaIndices(self) -> Type[UnwrapViaIndices]:
        return self.transformation_wrapper(UnwrapViaIndices)

    @property
    def VelocityFromPositions(self) -> Type[VelocityFromPositions]:
        return self.transformation_wrapper(VelocityFromPositions)

    #####################
    #### Calculators ####
    #####################

    @property
    def AngularDistributionFunction(self):
        return self._calculator_shim(AngularDistributionFunction)

    @property
    def CoordinationNumbers(self):
        return self._calculator_shim(CoordinationNumbers)

    @property
    def EinsteinDiffusionCoefficients(self):
        return self._calculator_shim(EinsteinDiffusionCoefficients)

    @property
    def EinsteinDistinctDiffusionCoefficients(self):
        return self._calculator_shim(EinsteinDistinctDiffusionCoefficients)

    @property
    def EinsteinHelfandIonicConductivity(self):
        return self._calculator_shim(EinsteinHelfandIonicConductivity)

    @property
    def EinsteinHelfandThermalKinaci(self):
        return self._calculator_shim(EinsteinHelfandThermalKinaci)

    @property
    def GreenKuboViscosityFlux(self):
        return self._calculator_shim(GreenKuboViscosityFlux)

    @property
    def GreenKuboDistinctDiffusionCoefficients(self):
        return self._calculator_shim(GreenKuboDistinctDiffusionCoefficients)

    @property
    def GreenKuboIonicConductivity(self):
        return self._calculator_shim(GreenKuboIonicConductivity)

    @property
    def GreenKuboDiffusionCoefficients(self):
        return self._calculator_shim(GreenKuboDiffusionCoefficients)

    @property
    def GreenKuboThermalConductivity(self):
        return self._calculator_shim(GreenKuboThermalConductivity)

    @property
    def GreenKuboViscosity(self):
        return self._calculator_shim(GreenKuboViscosity)

    @property
    def KirkwoodBuffIntegral(self):
        return self._calculator_shim(KirkwoodBuffIntegral)

    @property
    def PotentialOfMeanForce(self):
        return self._calculator_shim(PotentialOfMeanForce)

    @property
    def RadialDistributionFunction(self):
        return self._calculator_shim(RadialDistributionFunction)

    @property
    def EinsteinHelfandThermalConductivity(self):
        return self._calculator_shim(EinsteinHelfandThermalConductivity)

    @property
    def StructureFactor(self):
        return self._calculator_shim(StructureFactor)
