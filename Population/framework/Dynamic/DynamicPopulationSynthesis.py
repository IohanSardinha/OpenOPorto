"""Dynamic population synthesis contracts.

Dynamic synthesizers build on a static synthesizer and a demographic update
step to evolve a population over time.
"""

from __future__ import annotations

from typing import Optional

from ..PopulationSynthesis import PopulationSynthesis
from ..Static.StaticPopulationSynthesis import StaticPoplationSynthesis
from .Update import DemographicUpdater

class DynamicPopulationSynthesis(PopulationSynthesis):
    """Abstract base class for dynamic population synthesis workflows."""

    synthesizer: Optional[StaticPoplationSynthesis] = None
    demographic_updater: Optional[DemographicUpdater] = None