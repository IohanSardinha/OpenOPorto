"""Integerization helpers for synthetic reconstruction."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

class Integerizer(ABC):
    """Abstract base class for integerizing continuous reconstruction output.
    
    Call
    ----
    __call__(data: np.ndarray) -> tuple[np.ndarray, Any]
        Convert a continuous matrix into an integerized representation.
    """

    columns = []
    impossibilities = []
    @abstractmethod
    def __call__(self, data: np.ndarray) -> tuple[np.ndarray, Any]:
        """Convert a continuous matrix into an integerized representation.

        :param data: Continuous matrix to integerize.
        :type data: np.ndarray
        :returns: Integerized data and validation output.
        :rtype: tuple[np.ndarray, Any]
        """
        pass

class DefaultIntegerizer(Integerizer):
    """Default integerizer that rounds while preserving totals.
    
    Methods
    ----
    __call__(data: np.ndarray) -> tuple[np.ndarray, dict[str, float]]
        Convert a continuous matrix into an integerized representation.
    validate(data: np.ndarray) -> dict[str, float]
        Evaluate the integerized matrix against the continuous source.
    """
    
    def __init__(self, columns: list[Any] | None = None, impossibilities: list[Any] | None = None) -> None:
        """Store integerization dimensions and forbidden combinations.

        :param columns: Ordered category values per dimension.
        :type columns: list[Any] | None
        :param impossibilities: Forbidden category combinations.
        :type impossibilities: list[Any] | None
        :returns: ``None``.
        :rtype: None
        """
        columns = columns or []
        impossibilities = impossibilities or []
        self.columns = columns
        self.impossibilities = impossibilities

    def __setImpossiblesAsZeros(self, data: np.ndarray) -> np.ndarray:
        """Redistribute forbidden mass and set impossible cells to zero.

        :param data: Continuous matrix to adjust.
        :type data: np.ndarray
        :returns: Adjusted matrix.
        :rtype: np.ndarray
        """
        
        index_maps = [{v: i for i, v in enumerate(col)} for col in self.columns]

        
        forb_idx = tuple(
            np.array([
                [index_maps[d][v] for d, v in enumerate(tup)]
                for tup in self.impossibilities
            ]).T
        )

        forb_sum = data[forb_idx].sum()

        num_rest = data.size - len(self.impossibilities)
        if num_rest > 0:
            redistrib = forb_sum / num_rest
            data += redistrib

        data[forb_idx] = 0

        return data

    def __call__(self, data: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Integerize the provided matrix.

        :param data: Continuous matrix to integerize.
        :type data: np.ndarray
        :returns: Integerized matrix and validation metrics.
        :rtype: tuple[np.ndarray, dict[str, float]]
        """

        self.continuous = data.copy()

        #First redestribute evenly the impossibilities that may have gotten some residual value
        if len(self.impossibilities) > 0:
            data = self.__setImpossiblesAsZeros(data)

        #Then integerize them
        floors = np.floor(data)
        reminders = data - floors

        select = int(round(data.sum()) - floors.sum())

        top_idx = np.column_stack(np.unravel_index(np.argsort((-reminders).ravel()), reminders.shape))[:select]

        np.add.at(floors, tuple(top_idx.T), 1)

        return floors, self.validate(floors)
    
    def validate(self, data: np.ndarray) -> dict[str, float]:
        """Evaluate the integerized matrix against the continuous source.

        :param data: Integerized matrix to validate.
        :type data: np.ndarray
        :returns: Validation metrics.
        :rtype: dict[str, float]
        """
        
        diff = data - self.continuous
        rmse = np.sqrt(np.sum(diff ** 2) / diff.size)
        
        
        ftr = 4 * np.sum(
            (np.sqrt(data) - np.sqrt(self.continuous)) ** 2
        )
        
        return {
            "rmse": rmse,
            "ftr": ftr
        }

class GibbsIntegerizer(Integerizer):
    """Placeholder for a Gibbs-sampling based integerizer."""

    def __init__(self) -> None:
        """Create a Gibbs integerizer placeholder."""
        raise NotImplementedError("GibbsIntegerizer is not implemented yet")