"""Combined component synthesis strategies."""

from __future__ import annotations

from .MultipleComponentSynthesis import MultipleComponentSynthesis
from ...Component.ComponentSynthesis import CombinedCapable

class CombinedSynthesis(MultipleComponentSynthesis):
    """Placeholder for combined multi-component synthesis."""

    synthesizer: CombinedCapable = None
