from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict

"""Population synthesis base interfaces.

This module defines the abstract contract used by population synthesis
of the framework.
"""

class PopulationSynthesis(ABC):
    """Abstract base class for population synthesis workflows.

    Concrete implementations should provide a full lifecycle composed of
    loading input data, running synthesis logic, and exporting outputs.

    Methods
    -------
    load() -> None
        Load all required input data for synthesis.
    run() -> pd.DataFrame | Dict
        Execute the synthesis procedure.
    export() -> None
        Export synthesized population artifacts.
    """

    def load(self)-> None:
        """Load all required input data for synthesis.

        Subclasses can override this method to perform any setup steps
        needed before :meth:`run`.

        :returns: ``None``.
        :rtype: None
        """
        pass
    
    @abstractmethod
    def run(self)->pd.DataFrame|Dict:
        """Execute the synthesis procedure.

        Implementations must provide the core synthesis behavior.

        :returns: The synthesized population artifacts.
        :rtype: pd.DataFrame or Dict
        """
        pass

    def export(self)-> None:
        """Export synthesized population artifacts.

        Subclasses can override this method to persist generated outputs
        (for example, XML, CSV, or JSON files).

        :returns: ``None``.
        :rtype: None
        """
        pass

    def __str__(self):
        """Return a string identifier, used for hash when caching.

        :returns: The concrete class name.
        :rtype: str
        """
        return f"{self.__class__.__name__}"
