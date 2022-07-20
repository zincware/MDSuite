import pytest

from mdsuite.experiment.experiment import Experiment


def test_experiment_name():
    with pytest.raises(ValueError):
        Experiment(experiment_name="250K", project=None)
