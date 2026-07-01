from .MultipleComponentSynthesis import MultipleComponentSynthesis
from ...Component.ComponentSynthesis import CombinedCapable

class CombinedSynthesis(MultipleComponentSynthesis):
    synthesizer: CombinedCapable = None
    raise NotImplementedError("CombinedSynthesis is not implemented yet")