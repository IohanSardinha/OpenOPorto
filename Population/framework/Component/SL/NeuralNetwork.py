"""Neural network based synthesis contracts."""

from abc import ABC

from .StatisticalLearning import StatisticalLearning

class NeuralNetwork(StatisticalLearning, ABC):
    """Abstract base class for neural network synthesizers."""

    pass

class VAESynthesis(NeuralNetwork):
    """Placeholder for variational autoencoder based synthesis."""

    pass