import numpy as np
import pandas as pd
from random import choices
from itertools import product
from ...ProcessStep import ProcessStep
from ...universal.misc import JOIN_MODE, TripProfileBuilder

#Match the persons to the activities
class DefaultActivityMatcher(ProcessStep):
    
    def process(self, population, persons, joinOn, mapper={}, joinMode=JOIN_MODE.BOTH, populationReductionFactor=1, prioritizeWhenMissing=None):
        
        #Precompute before procedure

        profiles = TripProfileBuilder().build(persons,keyLabels=joinOn)

        population["ReducedValue"] = (population["value"]*populationReductionFactor).round().astype(int)

        #Trip profile key must match the population keys

        population = (population
                      .reset_index(drop=True)
                      .loc[lambda df: df.index.repeat(df["ReducedValue"])]
                      .drop(columns=["value", "ReducedValue"])
                      .reset_index(drop=True))

        matches = []
        cs = population.columns.tolist()
        joinon_idx = [cs.index(k) for k in joinOn]
        all_keys = list(profiles.keys()) #CHECK
        profilePool = {k: v.copy() for k, v in profiles.items()}

        compareList = lambda x,y: y in x if type(x) == list else x == y

        if joinMode in (JOIN_MODE.BOTH, JOIN_MODE.RIGHT):
            sample_attr_keys = next(iter(persons.values()))["attributes"].keys()
            include_attr_keys = [k for k in sample_attr_keys if k not in population.columns]

        self._match_stats = {"exact": 0, "prioritized": 0, "random": 0, "total": 0}

        #Procedure
        for person in population.itertuples(index=False, name=None):

            mapped_attributes = [mapper[person[key]] if person[key] in mapper else person[key] for key in joinon_idx ]
            mapped_attributes_list = [attribute if type(attribute) == list else [attribute] for attribute in mapped_attributes]
            keys = ["|".join(y) for y in product(*mapped_attributes_list) if "|".join(y) in profilePool]

            if not keys:
                if prioritizeWhenMissing:
                    attIndex = cs.index(prioritizeWhenMissing)
                    
                    mapped_attribute = mapper[person[attIndex]] if person[attIndex] in mapper else person[attIndex]
                    prioritizeWhenMissingIndex = joinOn.index(prioritizeWhenMissing)
                    keys = [k for k in profiles.keys() if compareList(mapped_attribute, k.split("|")[prioritizeWhenMissingIndex])]

                    if not keys:
                        keys = list(profiles.keys())
                        self._match_stats["random"] += 1
                    else:
                        self._match_stats["prioritized"] += 1
                else:
                    keys = list(profiles.keys())
                    self._match_stats["random"] += 1
            else:
                self._match_stats["exact"] += 1
            self._match_stats["total"] += 1

            if sum([len(profilePool[key]) for key in keys]) == 0:
                for key in keys:
                    profilePool[key] = profiles[key].copy()

            lengths = np.array([len(profilePool[k]) for k in keys])
            i_key = choices(keys, weights=lengths, k=1)[0]

            j_key = np.random.randint(len(profilePool[i_key]))

            trip = profilePool[i_key].pop(j_key)
            
            if joinMode in (JOIN_MODE.BOTH, JOIN_MODE.RIGHT):
                attr_dict = persons[trip]["attributes"]
                trip = [attr_dict[k] for k in include_attr_keys] + [trip]
            
            matches.append(trip)

        #Joining
        if joinMode == JOIN_MODE.LEFT:
            newCol = pd.DataFrame(matches, columns=["match"])
            population = population.join(newCol)
        elif joinMode == JOIN_MODE.BOTH:
            newCols = pd.DataFrame(matches, columns=[k for k in list(persons.values())[0]["attributes"].keys() if not k in population.columns]+["match"])
            population = population.join(newCols)
        else:
            population = pd.DataFrame(matches, columns=[k for k in list(persons.values())[0]["attributes"].keys() if not k in population.columns]+["match"])


        return population, self.validate(population, persons, joinOn)

    def validate(self, population, persons, joinOn):
        print(self._match_stats)
        total = self._match_stats["total"]
        
        match_quality = {
            "exact_rate":      self._match_stats["exact"]      / total,
            "prioritized_rate": self._match_stats["prioritized"] / total,
            "random_rate":     self._match_stats["random"]     / total,
        }
        
        # Distribution preservation: did matched attributes mirror input?
        distribution_rmse = {}
        for col in joinOn:
            if col in population.columns and col in persons:
                input_dist  = population[col].value_counts(normalize=True).sort_index()
                output_dist = population[col].value_counts(normalize=True).sort_index()
                diff = input_dist.subtract(output_dist, fill_value=0)
                distribution_rmse[col] = np.sqrt((diff ** 2).mean())
        
        return {
            "match_quality": match_quality,
            "distribution_rmse": distribution_rmse
        }