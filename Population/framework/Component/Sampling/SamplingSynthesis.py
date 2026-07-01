from abc import ABC, abstractmethod
from ..ComponentSynthesis import ComponentSynthesis
import random
import pandas as pd

class Sampling(ComponentSynthesis):
    def __init__(self, component, population, sample_size=1, mode="random"):
        self.component = component
        self.sample_size = sample_size
        self.population = population
        self.mode = mode

    def sample_random(self):
        if type(self.population) == pd.DataFrame:
            return self.population.sample(frac=self.sample_size, random_state=42).reset_index(drop=True)
        if type(self.population) == dict:
            keys = list(self.population.keys())
            sampled_keys = random.sample(keys, int(self.sample_size*len(keys)))
            return {key: self.population[key] for key in sampled_keys}
        if type(self.population) == set:
            return random.sample(sorted(self.population), int(self.sample_size*len(self.population)))
        return random.sample(self.population, int(self.sample_size*len(self.population)))

    def sample_iterative(self):
        raise NotImplementedError("Iterative sampling not implemented yet.")

    def sample(self, mode=None):
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
        
    def synthesize(self):
        return self.sample()

class ActivityChainSampler(Sampling):

    def validate_generic_format(self, population):
        return  (isinstance(population, dict) and  
                 all(isinstance(person, dict) for person in population.values()) and
                 all("attributes" in person and len(person["attributes"]) > 0 for person in population.values()) and
                 all(  "tripDesc" in person and len(person["tripDesc"]) > 0 for person in population.values()) and
                 all(      "legs" in person and len(person["legs"]) > 0 for person in population.values()))

    def __init__(self, population, sample_size=1, has_location=False, mode="random"):
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
   
        