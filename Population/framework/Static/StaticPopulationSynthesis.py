"""Static population synthesis contracts.

Static synthesizers produce a population without time evolution.
"""

from abc import ABC
from ..PopulationSynthesis import PopulationSynthesis

class StaticPoplationSynthesis(PopulationSynthesis, ABC):
    """Abstract base class for static population synthesis workflows."""

    pass