from abc import ABC
from ..StaticPopulationSynthesis import StaticPoplationSynthesis
from ...Component.ComponentSynthesis import ComponentSynthesis
from ...misc import cache

class MultipleComponentSynthesis(StaticPoplationSynthesis, ABC):

    def __init__(self, components: dict[str, ComponentSynthesis]):
        self.components = {k:components[k] for k in ComponentSynthesis.COMPONTENTS if k in components}
    
    def run_single(self, component: ComponentSynthesis):
        return component()
    
    def run(self):
        self.results = {}
        for name, component in self.components.items():
            self.results[name] = self.run_single(component)
        return self.results