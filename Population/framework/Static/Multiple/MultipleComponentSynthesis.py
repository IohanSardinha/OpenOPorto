"""Multiple component synthesis base classes."""

from abc import ABC
from typing import Any

from ..StaticPopulationSynthesis import StaticPoplationSynthesis
from ...Component.ComponentSynthesis import ComponentSynthesis

class MultipleComponentSynthesis(StaticPoplationSynthesis, ABC):
    """Abstract base class for synthesis workflows with multiple components.
    
    Methods
    -------
    run_single(component: ComponentSynthesis) -> Any
        Execute a single component synthesizer.
    run() -> dict[str, Any]
        Run all configured component synthesizers and return their results.
    """

    def __init__(self, components: dict[str, ComponentSynthesis]) -> None:
        """Store the component map used by the synthesis workflow.

        :param components: Mapping from component names to synthesizers.
        :type components: dict[str, ComponentSynthesis]
        :returns: ``None``.
        :rtype: None
        """
        self.components = {k: components[k] for k in ComponentSynthesis.COMPONTENTS if k in components}
    
    def run_single(self, component: ComponentSynthesis) -> Any:
        """Execute a single component synthesizer.

        :param component: Component synthesizer to run.
        :type component: ComponentSynthesis
        :returns: Component result.
        :rtype: Any
        """
        return component()
    
    def run(self) -> dict[str, Any]:
        """Run all configured component synthesizers.

        :returns: Mapping of component names to synthesis results.
        :rtype: dict[str, Any]
        """
        self.results = {}
        for name, component in self.components.items():
            self.results[name] = self.run_single(component)
        return self.results