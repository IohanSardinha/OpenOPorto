"""Synthetic reconstruction contracts.

Reconstruction-based synthesizers infer a synthetic component from observed
data and auxiliary constraints.
"""

from abc import ABC

from ..ComponentSynthesis import ComponentSynthesis

class SyntheticReconstruction(ComponentSynthesis, ABC):
    """Abstract base class for reconstruction-based synthesis."""

    pass