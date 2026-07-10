"""Merging strategies for multiple component synthesis."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from copy import deepcopy
from random import choices
from itertools import product, chain

from .MultipleComponentSynthesis import MultipleComponentSynthesis
from ...Component.ComponentSynthesis import ComponentSynthesis

class MergeSynthesis(MultipleComponentSynthesis, ABC):
    """Base class for synthesis strategies that merge multiple results.
    
    Methods
    -------
    merge(results: dict[Any, Any]) -> Any
        Combine multiple synthesis outputs into a single result.
    """

    @abstractmethod
    def merge(self, results: dict[Any, Any]) -> Any:
        """Combine multiple synthesis outputs into a single result.

        :param results: Results returned by the underlying component runs.
        :type results: dict[Any, Any]
        :returns: Merged synthesis output.
        :rtype: Any
        """
        if len(self.components) < 2:
            raise ValueError("MergeSynthesis requires at least two components to merge")

class PoolMatching():
    """Placeholder for pool-based matching strategies."""

    pass

class AttributeMatching(MergeSynthesis):
    """Match attribute records with activity or relation records.
    
    Methods
    -------
    run() -> Any
        Execute the component synthesizers and merge the results.
    """

    def __init__(self, components: dict[str, ComponentSynthesis], joinOn: list[str] | None = None, joinMode: int = 1, keyMapper: dict[Any, Any] | None = None, prioritizeWhenMissing: dict[Any, Any] | None = None) -> None:
        """Configure join keys and matching preferences.

        :param components: Mapping of component names to synthesizers.
        :type components: dict[str, ComponentSynthesis]
        :param joinOn: Fields used to build matching keys.
        :type joinOn: list[str] | None
        :param joinMode: Matching mode selector.
        :type joinMode: int
        :param keyMapper: Optional mapping applied before matching.
        :type keyMapper: dict[Any, Any] | None
        :param prioritizeWhenMissing: Fallback priorities when keys are missing.
        :type prioritizeWhenMissing: dict[Any, Any] | None
        :returns: ``None``.
        :rtype: None
        """
        super().__init__(components)
        keyMapper = keyMapper or {}
        prioritizeWhenMissing = prioritizeWhenMissing or {}
        joinOn.remove("gender") #Temporary - testing
        self.joinOn = joinOn
        self.joinMode = joinMode
        self.keyMapper = keyMapper
        self.prioritizeWhenMissing = prioritizeWhenMissing

    def run(self) -> Any:
        """Run the component synthesizers and merge the results.

        :returns: Merged synthesis output.
        :rtype: Any
        """
        super().run()
        self.results = self.merge(self.results)
        return self.results

    def __make_key(self, person: Any, acessor: Any) -> str:
        """Construct a matching key for a given record.

        :param person: Record to generate a key for.
        :type person: Any
        :param acessor: Function to access record fields.
        :type acessor: Any
        :returns: Concatenated key string.
        :rtype: str
        """

        return "|".join([acessor(person, key) for key in self.joinOn])

    def __get_mapped_keys(self, item: Any, accessor: Any, keyMapper: dict[Any, Any]) -> list[str]:
        """Generate a list of mapped keys for matching.

        :param item: Record to generate keys for.
        :type item: Any
        :param accessor: Function to access record fields.
        :type accessor: Any
        :param keyMapper: Mapping to apply to field values.
        :type keyMapper: dict[Any, Any]
        :returns: List of concatenated key strings.
        :rtype: list[str]
        """

        mapped_attributes = [keyMapper.get(accessor(item, key), accessor(item, key)) for key in self.joinOn]
        mapped_attributes_list = [attribute if type(attribute) == list else [attribute] for attribute in mapped_attributes]
        return ["|".join(p) for p in product(*mapped_attributes_list)]

    def __get_keys(self, pop: Any) -> list[str]:

        """Extract the list of keys from a population structure.

        :param pop: Population data structure.
        :type pop: Any
        :returns: List of keys present in the population.
        :rtype: list[str]
        """

        if type(pop) == pd.DataFrame:
            return list(pop.columns)
        elif type(pop) == dict:
            return list(list(pop.values())[0]["attributes"].keys())
        
    def __make_iterator(self, pop: Any):
        """Create an iterator over the population records.

        :param pop: Population data structure.
        :type pop: Any
        :returns: Iterator over population records.
        :rtype: Iterator
        """

        if type(pop) == pd.DataFrame:
            return pop.itertuples(index=False, name=None)
        elif type(pop) == dict:
            return pop.values()
        raise ValueError("Unsupported population type for matching")

    def __make_profile(self, pop: Any, accessor: Any, prioritizeWhenMissing: Any = None) -> tuple[dict[str, list[int]], dict[Any, list[str]]]:
        """Construct a profile mapping keys to record indices.

        :param pop: Population data structure.
        :type pop: Any
        :param accessor: Function to access record fields.
        :type accessor: Any
        :param prioritizeWhenMissing: Optional field for prioritization.
        :type prioritizeWhenMissing: Any
        :returns: Tuple containing the key-to-indices profile and the priority profile.
        :rtype: tuple[dict[str, list[int]], dict[Any, list[str]]]
        """
        
        profiles= {}
        priority_profile = {}
        for i, person in enumerate(self.__make_iterator(pop)):
            key = self.__make_key(person, accessor)
            profiles[key] = profiles.get(key, []) + [i]
            if prioritizeWhenMissing:
                priority_key = accessor(person, prioritizeWhenMissing)
                priority_profile[priority_key] = priority_profile.get(priority_key, []) + [key]
        return profiles, priority_profile
    
    def __make_idx_accessor(self, pop: Any)-> Any:
        """Create an accessor function to retrieve records by index.

        :param pop: Population data structure.
        :type pop: Any
        :returns: Function that retrieves a record given its index.
        :rtype: Callable[[int], Any]
        """
        if type(pop) == pd.DataFrame:
            return lambda idx: pop.iloc[idx]
        elif type(pop) == dict:
            values = list(pop.values())
            return lambda idx: values[idx]
        raise ValueError("Unsupported population type for matching")

    def __match_common(self, A: Any, B: Any, accessorA: Any, accessorB: Any, keyMapper: dict[Any, Any] | None = None, prioritizeWhenMissing: Any = None) -> np.ndarray:
        """Match rows between two populations using weighted key profiles.
        
        :param A: First population data structure.
        :type A: Any
        :param B: Second population data structure.
        :type B: Any
        :param accessorA: Function to access fields in population A.
        :type accessorA: Any
        :param accessorB: Function to access fields in population B.
        :type accessorB: Any
        :param keyMapper: Optional mapping applied before matching.
        :type keyMapper: dict[Any, Any] | None
        :param prioritizeWhenMissing: Optional field for prioritization.
        :type prioritizeWhenMissing: Any
        :returns: Array of indices in B matched to each record in A.
        :rtype: np.ndarray
        """

        keyMapper = keyMapper or {}
        matches = np.zeros((len(A)), dtype=int)
        profiles, priority_profile = self.__make_profile(B, accessorB, prioritizeWhenMissing)

        profilePool = deepcopy(profiles)

        for i, a in enumerate(self.__make_iterator(A)):
            possible_keys = self.__get_mapped_keys(a, accessorA, keyMapper)
            valid_keys = [key for key in possible_keys if key in profiles]
            if len(valid_keys) == 0:
                if prioritizeWhenMissing:
                    priority_values = keyMapper.get(accessorA(a, prioritizeWhenMissing), accessorA(a, prioritizeWhenMissing))
                    priority_values = priority_values if type(priority_values) == list else [priority_values]
                    valid_keys = []
                    for key in priority_values:
                        valid_keys += priority_profile.get(key, [])                    
            if len(valid_keys) == 0:
                valid_keys = list(profiles.keys())

            if sum([len(profilePool[key]) for key in valid_keys]) == 0:
                for key in valid_keys:
                    profilePool[key] = profiles[key].copy()

            lengths = np.array([len(profilePool[k]) for k in valid_keys])
            i_key = choices(valid_keys, weights=lengths, k=1)[0]

            j_key = np.random.randint(len(profilePool[i_key]))

            b_idx = profilePool[i_key].pop(j_key)
            matches[i] = b_idx

            if i%1000 == 0:
                print(f"Matching progress: {i/len(A)*100}%")

        return matches

    def __make_acessor(self, attributes: Any, path: list[str] | None = None)-> Any:
        """Create an accessor function for a given population structure.

        :param attributes: Population data structure.
        :type attributes: Any
        :param path: Optional path for nested access.
        :type path: list[str] | None
        :returns: Function that retrieves a field value given a record and key.
        :rtype: Callable[[Any, str], Any]
        """

        if type(attributes) == pd.DataFrame:
            column_index = {col: idx for idx, col in enumerate(attributes.columns)}
            return lambda p, k: p[column_index[k]]
        elif type(attributes) == dict:
            deep = lambda x, keys: x if len(keys) == 0 else deep(x[keys[0]], keys[1:])
            return lambda p, k: deep(p, path)[k] 

    def __merge_attributes_to_activities(self, attributes: Any, activities: Any):
        """Merge attribute records with activity records based on matching keys.

        :param attributes: Attribute population data structure.
        :type attributes: Any
        :param activities: Activity population data structure.
        :type activities: Any
        :returns: Merged population data structure.
        :rtype: Any
        """

        attributes_accessor = self.__make_acessor(attributes)
        activities_accessor = self.__make_acessor(activities, path=["attributes"])

        matches = self.__match_common(attributes, activities, attributes_accessor, activities_accessor, self.keyMapper.get(ComponentSynthesis.COMPONTENTS.Attributes, {}), self.prioritizeWhenMissing.get(ComponentSynthesis.COMPONTENTS.Attributes, None))

        attributes_accessor_idx = self.__make_idx_accessor(attributes)
        activities_accessor_idx = self.__make_idx_accessor(activities)
        
        attribute_keys = self.__get_keys(attributes)
        attribute_keys.remove("value")
        activity_keys = self.__get_keys(activities)
        for key in attribute_keys:
            if key in activity_keys:
                activity_keys.remove(key)

        max_legs = max([len(activities_accessor_idx(act_idx)["legs"]) for act_idx in matches])
        leg_size = len(activities_accessor_idx(matches[0])["legs"])

        results = []
        
        for att_idx, act_idx in enumerate(matches):
            person_attributes = attributes_accessor_idx(att_idx)
            person_activities = activities_accessor_idx(act_idx)
            result = [attributes_accessor(person_attributes, key) for key in attribute_keys] + \
                     [activities_accessor(person_activities, key) for key in activity_keys] + \
                     [len(person_activities["legs"])] + \
                     list(chain(*[leg.values() for leg in person_activities["legs"]])) + \
                     [None]*(max_legs - len(person_activities["legs"]))*leg_size
            results.append(result)
        
        columns = attribute_keys + activity_keys + ["leg_count"] + list(chain(*[["leg_"+str(i)+"_"+key for key in person_activities["legs"][0].keys()] for i in range(max_legs)]))
        return pd.DataFrame(results, columns=columns)
            

    def __merge_attributes_to_relations(self, attributes: Any, relations: Any):
        """Merge attribute records with relation records based on matching keys.

        :param attributes: Attribute population data structure.
        :type attributes: Any
        :param relations: Relation population data structure.
        :type relations: Any
        :returns: Merged population data structure.
        :rtype: Any
        """
        raise NotImplementedError("Merging attributes to relations is not yet implemented.")

    def __merge_activities_to_relations(self, activities: Any, relations: Any):
        """Merge activity records with relation records based on matching keys.

        :param activities: Activity population data structure.
        :type activities: Any
        :param relations: Relation population data structure.
        :type relations: Any
        :returns: Merged population data structure.
        :rtype: Any
        """
        raise NotImplementedError("Merging activities to relations is not yet implemented.")

    def merge(self, results: dict[Any, Any]):

        """Combine multiple synthesis outputs into a single result.

        :param results: Results returned by the underlying component runs.
        :type results: dict[Any, Any]
        :returns: Merged synthesis output.
        :rtype: Any
        """

        super().merge()
        
        #A-C
        if ComponentSynthesis.COMPONTENTS.Attributes in results and\
           ComponentSynthesis.COMPONTENTS.Activities in results:
            attribute_to_activities = self.__merge_attributes_to_activities(results[ComponentSynthesis.COMPONTENTS.Attributes],
                                                                     results[ComponentSynthesis.COMPONTENTS.Activities])
            if ComponentSynthesis.COMPONTENTS.Relations not in results:
                print("AAA")
                return attribute_to_activities
        #A-C-R
        if ComponentSynthesis.COMPONTENTS.Attributes in results and\
           ComponentSynthesis.COMPONTENTS.Activities in results and\
           ComponentSynthesis.COMPONTENTS.Relations in results:
           return self.__merge_attributes_to_relations(attribute_to_activities, results[ComponentSynthesis.COMPONTENTS.Relations]) 

        #A-R  
        if ComponentSynthesis.COMPONTENTS.Attributes in results and\
           ComponentSynthesis.COMPONTENTS.Relations in results:
            return self.__merge_attributes_to_relations(results[ComponentSynthesis.COMPONTENTS.Attributes],
                                                            results[ComponentSynthesis.COMPONTENTS.Relations])
        
        #C-R
        return self.__merge_activities_to_relations(results[ComponentSynthesis.COMPONTENTS.Activities],
                                                  results[ComponentSynthesis.COMPONTENTS.Relations])            

class SimulationMatching(MergeSynthesis):
    """Placeholder for simulation-based matching synthesis."""

    def __init__(self) -> None:
        """Create a simulation matching synthesizer placeholder."""
        pass