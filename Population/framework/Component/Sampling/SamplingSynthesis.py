"""Sampling based component synthesis strategies."""

from __future__ import annotations

from abc import ABC
from typing import Any

from ..ComponentSynthesis import ComponentSynthesis
import random
import pandas as pd

class Sampling(ComponentSynthesis):
    """Base class for sampling a population into a smaller subset.
    
    Methods
    -------
    synthesize()
        Execute the sampling workflow.
    """

    def __init__(self, component: ComponentSynthesis.COMPONTENTS, population: Any, sample_size: float = 1, mode: str = "random") -> None:
        """Store the sampling configuration.

        :param component: Component handled by the sampling synthesizer.
        :type component: ComponentSynthesis.COMPONTENTS
        :param population: Population to sample from.
        :type population: Any
        :param sample_size: Fraction or count used for sampling.
        :type sample_size: float
        :param mode: Sampling mode name.
        :type mode: str
        :returns: ``None``.
        :rtype: None
        """
        self.component = component
        self.sample_size = sample_size
        self.population = population
        self.mode = mode

    def sample_random(self) -> Any:
        """Return a random sample from the configured population.

        :returns: Sampled population subset.
        :rtype: Any
        """
        if type(self.population) == pd.DataFrame:
            return self.population.sample(frac=self.sample_size, random_state=42).reset_index(drop=True)
        if type(self.population) == dict:
            keys = list(self.population.keys())
            sampled_keys = random.sample(keys, int(self.sample_size*len(keys)))
            return {key: self.population[key] for key in sampled_keys}
        if type(self.population) == set:
            return random.sample(sorted(self.population), int(self.sample_size*len(self.population)))
        return random.sample(self.population, int(self.sample_size*len(self.population)))

    def sample_iterative(self) -> Any:
        """Return an iterative sample.

        :returns: Sampled population subset.
        :rtype: Any
        """
        raise NotImplementedError("Iterative sampling not implemented yet.")

    def sample(self, mode: str | None = None) -> Any:
        """Sample the population using the requested mode.

        :param mode: Optional sampling mode override.
        :type mode: str | None
        :returns: Sampled population subset.
        :rtype: Any
        """
        if self.sample_size == 1:
            return self.population
        if mode is None:
            mode = self.mode
        if mode == "random":
            return self.sample_random()
        elif mode == "iterative":
            return self.sample_iterative()
        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        
    def synthesize(self) -> Any:
        """Execute the sampling workflow.

        :returns: Sampled population subset.
        :rtype: Any
        """
        return self.sample()

class ActivityChainSampler(Sampling):
    """Sample activity chains from a population with validated structure."""

    def validate_generic_format(self, population: Any) -> bool:
        """Validate the expected generic activity-chain population format.

        :param population: Population object to validate.
        :type population: Any
        :returns: ``True`` when the population matches the expected format.
        :rtype: bool
        """
        return  (isinstance(population, dict) and  
                 all(isinstance(person, dict) for person in population.values()) and
                 all("attributes" in person and len(person["attributes"]) > 0 for person in population.values()) and
                 all(  "tripDesc" in person and len(person["tripDesc"]) > 0 for person in population.values()) and
                 all(      "legs" in person and len(person["legs"]) > 0 for person in population.values()))

    def __init__(self, population: Any, sample_size: float = 1, has_location: bool = False, mode: str = "random") -> None:
        """Create an activity chain sampler.

        :param population: Population to sample.
        :type population: Any
        :param sample_size: Fraction or count used for sampling.
        :type sample_size: float
        :param has_location: Whether legs already include coordinates.
        :type has_location: bool
        :param mode: Sampling mode name.
        :type mode: str
        :returns: ``None``.
        :rtype: None
        """
        if not self.validate_generic_format(population):
            raise ValueError("Invalid population format for ActivityChainSampler.")
        super().__init__(component=self.COMPONTENTS.Activities, population=population, sample_size=sample_size, mode=mode)
        self.has_location = has_location

        if not self.has_location:
            for id in self.population.keys():
                for i in range(len(self.population[id]["legs"])):
                    for key in ["x", "y"]:
                            if not key in self.population[id]["legs"][i]:
                                self.population[id]["legs"][i][key] = None
   
        