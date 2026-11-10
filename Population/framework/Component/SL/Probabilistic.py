"""Probabilistic synthesis contracts."""

import cloudpickle
import numpy as np
from abc import ABC
import pandas as pd
import bnlearn as bn
from enum import Enum
import multiprocessing as mp
from itertools import product
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed


from .StatisticalLearning import StatisticalLearning

class Probabilistic(StatisticalLearning, ABC):
    """Abstract base class for probabilistic learning synthesizers."""
    pass

class STRUCTURE(Enum):
    LEARN = 1
class PARAMETERS(Enum):
    LEARN = 1

class BayesianNetworkSynthesis(Probabilistic):

    def __init__(self, components, data, size, structure=STRUCTURE.LEARN,structure_args={}, parameters=PARAMETERS.LEARN, parameters_args={}, impossible=[], sampling_args={}):
        super().__init__(components)

        bn.set_logger("error")

        self.data = data
        self.size = size
        self.structure = structure
        self.structure_args = structure_args
        self.parameters = parameters
        self.parameters_args = parameters_args
        self.impossible = impossible
        self.sampling_args = sampling_args

    def from_csv(components, path, size, structure=STRUCTURE.LEARN,structure_args={}, parameters=PARAMETERS.LEARN, parameters_args={}, impossible=[], sampling_args={}):
        data = pd.read_csv(path)
        return BayesianNetworkSynthesis(components, data, size, structure, structure_args, parameters, parameters_args, impossible, sampling_args)

    def __learn_structure(self):
        return bn.structure_learning.fit(self.data, **self.structure_args)

    def __learn_parameters(self):
        return bn.parameter_learning.fit(self.model, self.data, **self.parameters_args)

    def __check_impossibilities(df, impossibilities):
        invalid = np.zeros(len(df), dtype=bool)

        for impossibility in impossibilities:
            mask = np.ones(len(df), dtype=bool)

            for col, values in impossibility.items():
                mask &= df[col].isin(values).to_numpy()

            invalid |= mask

        return df.iloc[np.flatnonzero(~invalid)].copy()

    def sample(self, size, evidence=None):
        accepted = []
        accepted_count = 0
        tries = 0
        while accepted_count < size:
            remaining = size - accepted_count
    
            pop = bn.sampling(self.model, 
                              n=remaining, 
                              methodtype=self.sampling_args.get('methodtype', 'bayes'), 
                              evidence=evidence, 
                              do=self.sampling_args.get('do', None), 
                              seed=self.sampling_args.get('seed', 0))
    
            valid = BayesianNetworkSynthesis.__check_impossibilities(pop, self.impossible)
            accepted.append(valid)
            accepted_count += len(valid)
            tries += 1
            self.sampling_args['seed'] = self.sampling_args.get('seed', 0) + 1
    
        result = pd.concat(accepted, ignore_index=True)
        return result.iloc[:size].copy()

    def synthesize(self):
        if self.structure == STRUCTURE.LEARN:
            self.model = self.__learn_structure()
        else:
            self.model = self.structure
            raise NotImplementedError("Framework currently only work by learning the structure")

        if self.parameters == PARAMETERS.LEARN:
            self.model = self.__learn_parameters()
        else:
            self.model.parameters = self.parameters
            raise NotImplementedError("Framework currently only work by learning the parameters")

        return self.sample(self.size, self.sampling_args.get('evidence', None))

class BayesianNetworkSynthesisWithEvidence(BayesianNetworkSynthesis):

    def __init__(self, components, data, reduction_factor, structure=STRUCTURE.LEARN,structure_args={}, parameters=PARAMETERS.LEARN, parameters_args={}, impossible=[], sampling_args={}, evidenceData=None, evidenceMapper={}):
        super().__init__(components, data, reduction_factor, structure, structure_args, parameters, parameters_args, impossible, sampling_args)
        self.evidenceData = evidenceData
        self.evidenceMapper = evidenceMapper

    def from_csv(components, path, reduction_factor, structure=STRUCTURE.LEARN,structure_args={}, parameters=PARAMETERS.LEARN, parameters_args={}, impossible=[], sampling_args={}, evidenceData=None, evidenceMapper={}):
        data = pd.read_csv(path)
        evidenceData = pd.read_csv(evidenceData) if evidenceData is not None else None
        return BayesianNetworkSynthesisWithEvidence(components, data, reduction_factor, structure, structure_args, parameters, parameters_args, impossible, sampling_args, evidenceData, evidenceMapper)

    def __expand_dict(d):
        keys = d.keys()
        values = [v if isinstance(v, list) else [v] for v in d.values()]
        return [dict(zip(keys, combination))for combination in product(*values)]

    def _sample_iteration(self, row):
        bn.set_logger("error")
        pops = []
        evidence = {}
        for var, maps in self.evidenceMapper.items():
            evidence[var] = maps[row[var]] if row[var] in maps else row[var]
        evidences = BayesianNetworkSynthesisWithEvidence.__expand_dict(evidence)
        n = (row["count"]*self.size)//len(evidences)
        if n==0: return pops
        for ev in evidences:
            pop = super().sample(int(n), evidence=ev)
            pop["section"] = row["section"]
            pops.append(pop)
        return pops
        

    def sample(self, _, __):
        pops = []

        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:

            futures = {
                executor.submit(self._sample_iteration, row): row
                for _, row in self.evidenceData.iterrows()
            }

            completed = 0
            for future in as_completed(futures):
                completed += 1
                print(f"Bayesian Network Sampling {completed*100/len(futures):.2f}%")
                try:
                    result = future.result()
                    pops.extend(result)

                except Exception as e:
                    row = futures[future]
                    print(f"Error processing row {row}: {e}")

        if not pops:
            return pd.DataFrame()

        return pd.concat(pops, ignore_index=True)