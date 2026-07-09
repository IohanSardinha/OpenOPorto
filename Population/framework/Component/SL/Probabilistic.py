"""Probabilistic synthesis contracts."""

from abc import ABC

from .StatisticalLearning import StatisticalLearning

class Probabilistic(StatisticalLearning, ABC):
    """Abstract base class for probabilistic learning synthesizers."""

    pass

class BayesianNetworkSynthesis(Probabilistic):
    """Placeholder for Bayesian network based synthesis."""

    pass
