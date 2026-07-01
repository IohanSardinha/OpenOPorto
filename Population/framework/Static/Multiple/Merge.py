from .MultipleComponentSynthesis import MultipleComponentSynthesis
from ...Component.ComponentSynthesis import ComponentSynthesis
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from copy import deepcopy
from random import choices
from itertools import product, chain
from ...misc import cache 

class MergeSynthesis(MultipleComponentSynthesis, ABC):
    @abstractmethod
    def merge(self):
        if len(self.components) < 2:
            raise ValueError("MergeSynthesis requires at least two components to merge")

class PoolMatching():
    pass

class AttributeMatching(MergeSynthesis):

    def __init__(self, components, joinOn=None, joinMode=1, keyMapper={}, prioritizeWhenMissing={}):
        super().__init__(components)
        joinOn.remove("gender") #Temporary - testing
        self.joinOn = joinOn
        self.joinMode = joinMode
        self.keyMapper = keyMapper
        self.prioritizeWhenMissing = prioritizeWhenMissing

    def run(self):
        super().run()
        self.results = self.merge(self.results)
        return self.results

    def __make_key(self, person, acessor):
        return "|".join([acessor(person, key) for key in self.joinOn])

    def __get_mapped_keys(self, item, accessor, keyMapper):
        mapped_attributes = [keyMapper.get(accessor(item, key), accessor(item, key)) for key in self.joinOn]
        mapped_attributes_list = [attribute if type(attribute) == list else [attribute] for attribute in mapped_attributes]
        return ["|".join(p) for p in product(*mapped_attributes_list)]

    def __get_keys(self, pop):
        if type(pop) == pd.DataFrame:
            return list(pop.columns)
        elif type(pop) == dict:
            return list(list(pop.values())[0]["attributes"].keys())
        
    def __make_iterator(self, pop):
        if type(pop) == pd.DataFrame:
            return pop.itertuples(index=False, name=None)
        elif type(pop) == dict:
            return pop.values()
        raise ValueError("Unsupported population type for matching")

    def __make_profile(self, pop, accessor, prioritizeWhenMissing=None):
        profiles= {}
        priority_profile = {}
        for i, person in enumerate(self.__make_iterator(pop)):
            key = self.__make_key(person, accessor)
            profiles[key] = profiles.get(key, []) + [i]
            if prioritizeWhenMissing:
                priority_key = accessor(person, prioritizeWhenMissing)
                priority_profile[priority_key] = priority_profile.get(priority_key, []) + [key]
        return profiles, priority_profile
    
    def __make_idx_accessor(self, pop):
        if type(pop) == pd.DataFrame:
            return lambda idx: pop.iloc[idx]
        elif type(pop) == dict:
            values = list(pop.values())
            return lambda idx: values[idx]
        raise ValueError("Unsupported population type for matching")

    def __match_common(self, A, B, accessorA, accessorB, keyMapper={}, prioritizeWhenMissing=None):
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

    def __make_acessor(self, attributes, path=None):
        if type(attributes) == pd.DataFrame:
            column_index = {col: idx for idx, col in enumerate(attributes.columns)}
            return lambda p, k: p[column_index[k]]
        elif type(attributes) == dict:
            deep = lambda x, keys: x if len(keys) == 0 else deep(x[keys[0]], keys[1:])
            return lambda p, k: deep(p, path)[k] 

    def __merge_attributes_to_activities(self, attributes, activities):
        
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
            

    def __merge_attributes_to_relations(self, attributes, relations):
        pass

    def __merge_activities_to_relations(self, activities, relations):
        pass

    def merge(self, results):
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
    def __init__(self):
        raise NotImplementedError("SimulationMatching is not implemented yet")