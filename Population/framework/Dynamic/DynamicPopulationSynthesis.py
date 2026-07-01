from ..PopulationSynthesis import PopulationSynthesis
from ..Static.StaticPopulationSynthesis import StaticPoplationSynthesis
from .Update import DemographicUpdater

class DynamicPopulationSynthesis(PopulationSynthesis):
    synthesizer: StaticPoplationSynthesis = None 
    demographic_updater: DemographicUpdater = None
    raise NotImplementedError("DynamicPopulationSynthesis is not implemented yet")