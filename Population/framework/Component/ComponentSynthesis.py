"""Base component synthesis contracts.

This module defines the common lifecycle hooks used by component-level
synthesis strategies.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

class ComponentSynthesis(ABC):
    """Abstract base class for component synthesis strategies.

    Concrete implementations should override :meth:`synthesize` and may
    optionally customize :meth:`before` and :meth:`after` hooks.

    Methods
    -------
    before() -> None
        Pre-synthesis setup logic.
    synthesize() -> Any
        Execute the component synthesis workflow.
    after(result: Any) -> tuple[bool, Any] | None
        Post-process the synthesis result.
    """

    COMPONTENTS = Enum("COMPONTENTS", ("Attributes", "Relations", "Activities"))

    def __init__(self, component: COMPONTENTS) -> ComponentSynthesis:
        """Store the component type handled by this synthesizer.

        :param component: Component category handled by the synthesizer.
        :type component: COMPONTENTS
        :returns: ComponentSynthesis instance.
        :rtype: ComponentSynthesis
        """
        self.component = component

    def before(self) -> None:
        """Run pre-synthesis setup logic.

        :returns: ``None``.
        :rtype: None
        """
        pass
    
    def after(self, result: Any) -> tuple[bool, Any] | None:
        """Post-process the synthesis result.

        :param result: Raw result returned by :meth:`synthesize`.
        :type result: Any
        :returns: Optional tuple indicating whether to replace the result.
        :rtype: tuple[bool, Any] | None
        """
        pass

    @abstractmethod
    def synthesize(self) -> Any:
        """Execute the component synthesis workflow.

        :returns: Synthesized component output.
        :rtype: Any
        """
        pass

    def __parse_after(self, result: Any, after_result: tuple[bool, Any] | None) -> Any:
        """Resolve the final output from the optional post-processing hook.

        :param result: Original synthesis result.
        :type result: Any
        :param after_result: Value returned by :meth:`after`.
        :type after_result: tuple[bool, Any] | None
        :returns: Final result to expose to callers.
        :rtype: Any
        """
        if after_result is not None:
            if isinstance(after_result, tuple) and len(after_result) == 2 and after_result[0] is True:
                return after_result[1]
        return result

    def __call__(self) -> Any:
        """Run the full synthesis lifecycle.

        :returns: Final synthesis result.
        :rtype: Any
        """
        self.before()
        result = self.synthesize()
        after_result = self.after(result)
        return self.__parse_after(result, after_result)

    def __str__(self) -> str:
        """Return a readable identifier for the component synthesizer.

        :returns: Concrete class name and component name.
        :rtype: str
        """
        return f"{self.__class__.__name__}|{self.component.name}"


class CombinedCapable(ComponentSynthesis, ABC):
    """Base class for synthesizers that combine multiple components."""

    def __init__(self, components: set[ComponentSynthesis.COMPONTENTS]) -> None:
        """Store the set of components handled by the synthesizer.

        :param components: Components that can be synthesized together.
        :type components: set[ComponentSynthesis.COMPONTENTS]
        :returns: ``None``.
        :rtype: None
        """
        self.components = components