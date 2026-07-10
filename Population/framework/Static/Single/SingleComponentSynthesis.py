"""Single component synthesis strategies."""

from __future__ import annotations

from typing import Any

from ..StaticPopulationSynthesis import StaticPoplationSynthesis
from ...Component import ComponentSynthesis

class SingleComponentSynthesis(StaticPoplationSynthesis):
    """Run a single component synthesizer.
    
    Methods
    -------
    run() -> Any
        Execute the wrapped component synthesizer and return its result.
    """

    def __init__(self, component_synthesis: ComponentSynthesis) -> None:
        """Store the component synthesizer to execute.

        :param component_synthesis: Component synthesizer instance.
        :type component_synthesis: ComponentSynthesis
        :returns: ``None``.
        :rtype: None
        """
        self.component_synthesis = component_synthesis

    def run(self) -> Any:
        """Execute the wrapped component synthesizer.

        :returns: Component synthesis result.
        :rtype: Any
        """
        return self.component_synthesis()

class ChainedSingleComponentSynthesis(StaticPoplationSynthesis):
    """Execute a chain of component synthesizers with argument threading.
    
    Methods
    -------
    run() -> Any
        Execute the chained synthesis workflow and return the final result.
    """
    
    class Future:
        """Reference to a future component result or attribute."""

        def __init__(self, att: Any) -> None:
            self.att = att

        def __str__(self) -> str:
            return self.att

        def __int__(self) -> int:
            return self.att
    class FutureResult(Future):
        """Placeholder resolved from a previous result."""

        pass
    class FutureComponent(Future):
        """Placeholder resolved from a previous component instance."""

        pass

    def __init__(self, parts: dict[ComponentSynthesis, tuple]) -> None:
        """Store the chained component definitions.

        :param parts: Mapping of component syntheses to their argument tuples.
        :type parts: dict[ComponentSynthesis, tuple]
        :returns: ``None``.
        :rtype: None
        """
        self.parts = parts
        self.components = []
    
    def __parse_args(self, args: tuple, prevResult: Any, prevComponent: Any) -> tuple:
        """Resolve future references in the argument list.

        :param args: Raw argument tuple for the next component.
        :type args: tuple
        :param prevResult: Result returned by the previous component.
        :type prevResult: Any
        :param prevComponent: Previously instantiated component.
        :type prevComponent: Any
        :returns: Resolved argument tuple.
        :rtype: tuple
        """
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, self.FutureResult):
                if isinstance(prevResult, tuple) or isinstance(prevResult, list):
                    args[i] = prevResult[int(arg)]
                else:
                    args[i] = getattr(prevResult, str(arg))
            if isinstance(arg,self.FutureComponent):
                args[i] = getattr(prevComponent, str(arg))

        return tuple(args)
    
    def run(self) -> Any:
        """Run the chained synthesis pipeline.

        :returns: Final component synthesis result.
        :rtype: Any
        """
        prevResult = None
        prevComponent = None
        for part, args in self.parts.items():
            prevComponent = part(*self.__parse_args(args, prevResult, prevComponent))
            self.components.append(prevComponent)
            synthesis = SingleComponentSynthesis(prevComponent)
            prevResult = synthesis.run()
        return prevResult