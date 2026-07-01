from abc import ABC
from .StatisticalLearning import StatisticalLearning

class NeuralNetwork(StatisticalLearning, ABC):
    pass

class VAESynthesis(NeuralNetwork):
    raise NotImplementedError("VAESynthesis is not implemented yet")