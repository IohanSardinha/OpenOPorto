from abc import ABC, abstractmethod

class PopulationSynthesis(ABC):
    pass

class StaticPoplationSynthesis(PopulationSynthesis, ABC):
    pass

class DemographicUpdater(ABC):
    pass

class Resampler(DemographicUpdater):
    pass

class LifeCourseSimulation(DemographicUpdater):
    pass

class AttributeProjector(DemographicUpdater):
    pass

class DynamicPopulationSynthesis(PopulationSynthesis):
    synthesizer: StaticPoplationSynthesis = None 
    demographic_updater: DemographicUpdater = None

class ComponentSynthesis(ABC):
    pass

class CombinedCapable(ComponentSynthesis, ABC):
    pass

class SyntheticReconstruction(ComponentSynthesis, ABC):
    pass

class IPFProcesser(ABC):
    pass

class IPF2DProcess(IPFProcesser):
    pass

class IPFHighDimProcess(IPFProcesser):
    pass

class Integerizer(ABC):
    pass

class DefaultIntegerizer(Integerizer):
    pass

class GibbsIntegerizer(Integerizer):
    pass

class IPFSinthesis(SyntheticReconstruction):
    processer: IPFProcesser = None
    integerizer: Integerizer = None

class IPUSinthesis(SyntheticReconstruction):
    pass

class CombinatorialOptimization(ComponentSynthesis, ABC):
    pass

class StatisticalLearning(CombinedCapable, ABC):
    pass

class Probabilistic(StatisticalLearning, ABC):
    pass

class NeuralNetwork(StatisticalLearning, ABC):
    pass

class BayesianNetworkSynthesis(Probabilistic):
    pass

class VAESynthesis(NeuralNetwork):
    pass

class Sampling(ComponentSynthesis, ABC):
    pass

class SingleComponentSynthesis(StaticPoplationSynthesis):
    synthesizer: ComponentSynthesis = None

class MultipleComponentSynthesis(StaticPoplationSynthesis, ABC):
    pass

class MergeSynthesis(MultipleComponentSynthesis, ABC):
    components: list[ComponentSynthesis] = None

class AttributeMatching(MergeSynthesis):
    pass

class SimulationMatching(MergeSynthesis):
    pass

class CombinedSynthesis(MultipleComponentSynthesis):
    synthesizer: CombinedCapable = None