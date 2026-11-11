"""Probabilistic synthesis contracts."""

import cloudpickle
import numpy as np
from abc import ABC
import pandas as pd
import bnlearn as bn
from enum import Enum
from typing import Any
import multiprocessing as mp
from itertools import product
from framework.Component.ComponentSynthesis import COMPONTENTS
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor, as_completed


from .StatisticalLearning import StatisticalLearning

class Probabilistic(StatisticalLearning, ABC):
    """Abstract base class for probabilistic learning synthesizers."""
    pass

class STRUCTURE(Enum):
    """Enum for Bayesian network structure learning\nIndicates that the structure should be learned from the data, and is not provided by the user."""
    LEARN = 1

class PARAMETERS(Enum):
    """Enum for Bayesian network parameter learning\nIndicates that the parameters should be learned from the data, and is not provided by the user."""
    LEARN = 1

class BayesianNetworkSynthesis(Probabilistic):

    """Bayesian Network Synthesizer
    
    Methods:
    -------
    from_csv: Create an instance from a CSV file.
    synthesize: Main method to synthesize a population based on the learned model.
    """

    def __init__(self, components:COMPONTENTS, data:pd.DataFrame, size:int, structure:STRUCTURE|str=STRUCTURE.LEARN,structure_args:dict={}, parameters:PARAMETERS|Any=PARAMETERS.LEARN, parameters_args:dict={}, impossible:list=[], sampling_args:dict={}):
        """Configure the Bayesian Network Synthesizer with the given parameters.

        :param components: The components to be synthesized.
        type components: self.COMPONTENTS
        :param data: The input data for learning the Bayesian network.
        type data: pd.DataFrame
        :param size: The desired size of the synthesized population.
        type size: int
        :param structure: The structure of the Bayesian network, either to be learned or provided.
        type structure: STRUCTURE|str
        :param structure_args: Additional arguments for structure learning.
        type structure_args: dict
        :param parameters: The parameters of the Bayesian network, either to be learned or provided.
        type parameters: PARAMETERS|Any
        :param parameters_args: Additional arguments for parameter learning.
        type parameters_args: dict
        :param impossible: List of impossible combinations to be filtered out.
        type impossible: list
        :param sampling_args: Additional arguments for the sampling process.
        type sampling_args: dict
        """
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

    def from_csv(components:COMPONTENTS, path:str, size:int, structure:STRUCTURE|str=STRUCTURE.LEARN,structure_args:dict={}, parameters:PARAMETERS|Any=PARAMETERS.LEARN, parameters_args:dict={}, impossible:list=[], sampling_args:dict={}):
        """
        Create an instance of BayesianNetworkSynthesis from a CSV file.
        
        :param components: The components to be synthesized.
        :type components: self.COMPONTENTS
        :param path: Path to the CSV file containing the data.
        :type path: str
        :param size: The desired size of the synthesized population.
        :type size: int
        :param structure: The structure of the Bayesian network, either to be learned or provided.
        :type structure: self.STRUCTURE | str
        :param structure_args: Additional arguments for structure learning.
        :type structure_args: dict
        :param parameters: The parameters of the Bayesian network, either to be learned or provided.
        :type parameters: self.PARAMETERS | Any
        :param parameters_args: Additional arguments for parameter learning.
        :type parameters_args: dict
        :param impossible: List of impossible combinations to be filtered out.
        :type impossible: list
        :param sampling_args: Additional arguments for the sampling process.
        :type sampling_args: dict
        :return: An instance of BayesianNetworkSynthesis.
        """
        data = pd.read_csv(path)
        return BayesianNetworkSynthesis(components, data, size, structure, structure_args, parameters, parameters_args, impossible, sampling_args)

    def __learn_structure(self):
        return bn.structure_learning.fit(self.data, **self.structure_args)

    def __learn_parameters(self):
        return bn.parameter_learning.fit(self.model, self.data, **self.parameters_args)

    def __check_impossibilities(df, impossibilities:list):

        """Check for impossible combinations in the DataFrame and filter them out.
        
        :param df: The DataFrame to be checked.
        :type df: pd.DataFrame
        :param impossibilities: A list of dictionaries, each representing an impossible combination of column values
        :type impossibilities: list
        """

        invalid = np.zeros(len(df), dtype=bool)

        for impossibility in impossibilities:
            mask = np.ones(len(df), dtype=bool)

            for col, values in impossibility.items():
                mask &= df[col].isin(values).to_numpy()

            invalid |= mask

        return df.iloc[np.flatnonzero(~invalid)].copy()

    def sample(self, size: int, evidence: dict=None):
        
        """Sample from the Bayesian network, filtering out impossible combinations.

        :param size: The number of samples to generate.
        :type size: int
        :param evidence: Optional evidence to condition the sampling on.
        :type evidence: dict, optional
        :return: A DataFrame containing the sampled data.
        """

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
        """Synthesize a population based on the learned model."""
        if self.structure == STRUCTURE.LEARN:
            self.model = self.__learn_structure()
        else:
            self.model = bn.import_DAG(self.structure)

        if self.parameters == PARAMETERS.LEARN:
            self.model = self.__learn_parameters()
        else:
            self.model.parameters = self.parameters
            raise NotImplementedError("Framework currently only work by learning the parameters")

        return self.sample(self.size, self.sampling_args.get('evidence', None))

class BayesianNetworkSynthesisWithEvidence(BayesianNetworkSynthesis):

    """Bayesian Network Synthesizer with Evidence Handling
    
    This class extends the BayesianNetworkSynthesis class to handle evidence during the sampling process. It allows for the specification of evidence data and a mapping for evidence variables, enabling more controlled and conditional sampling from the Bayesian network.

    Methods:
    -------
    from_csv: Create an instance from a CSV file, including evidence data and mapping.
    sample: Sample from the Bayesian network, considering the provided evidence and filtering out impossible combinations.
    """

    def __init__(self, components: list, data: pd.DataFrame, reduction_factor: float, structure:STRUCTURE|str=STRUCTURE.LEARN,structure_args:dict={}, parameters:PARAMETERS|Any=PARAMETERS.LEARN, parameters_args:dict={}, impossible:list=[], sampling_args:dict={}, evidenceData:pd.DataFrame=None, evidenceMapper:dict={}):

        """Configure the Bayesian Network Synthesizer with Evidence Handling.
        
        :param components: The components to be synthesized.
        :type components: list
        :param data: The input data for learning the Bayesian network.
        :type data: pd.DataFrame
        :param reduction_factor: The factor by which to reduce the sample size during synthesis.
        :type reduction_factor: float
        :param structure: The structure of the Bayesian network, either to be learned or provided.
        :type structure: STRUCTURE|str
        :param structure_args: Additional arguments for structure learning.
        :type structure_args: dict
        :param parameters: The parameters of the Bayesian network, either to be learned or provided.
        :type parameters: PARAMETERS|Any
        :param parameters_args: Additional arguments for parameter learning.
        :type parameters_args: dict
        :param impossible: List of impossible combinations to be filtered out.
        :type impossible: list
        :param sampling_args: Additional arguments for the sampling process.
        :type sampling_args: dict
        :param evidenceData: DataFrame containing evidence data for conditional sampling.
        :type evidenceData: pd.DataFrame, optional
        :param evidenceMapper: A mapping of evidence variables to their corresponding values.
        :type evidenceMapper: dict, optional
        """
        
        super().__init__(components, data, reduction_factor, structure, structure_args, parameters, parameters_args, impossible, sampling_args)
        self.evidenceData = evidenceData
        self.evidenceMapper = evidenceMapper

    def from_csv(components: list, path: str, reduction_factor: float, structure:STRUCTURE|str=STRUCTURE.LEARN,structure_args:dict={}, parameters: PARAMETERS=PARAMETERS.LEARN, parameters_args:dict={}, impossible:list=[], sampling_args:dict={}, evidenceData: pd.DataFrame=None, evidenceMapper: dict={}):
        """Create an instance of BayesianNetworkSynthesisWithEvidence from a CSV file.
        
        :param components: The components to be synthesized.
        :type components: list
        :param path: Path to the CSV file containing the data.
        :type path: str
        :param reduction_factor: The factor by which to reduce the sample size during synthesis.
        :type reduction_factor: float
        :param structure: The structure of the Bayesian network, either to be learned or provided.
        :type structure: STRUCTURE|str
        :param structure_args: Additional arguments for structure learning.
        :type structure_args: dict
        :param parameters: The parameters of the Bayesian network, either to be learned or provided.
        :type parameters: PARAMETERS|Any
        :param parameters_args: Additional arguments for parameter learning.
        :type parameters_args: dict
        :param impossible: List of impossible combinations to be filtered out.
        :type impossible: list
        :param sampling_args: Additional arguments for the sampling process.
        :type sampling_args: dict
        :param evidenceData: DataFrame containing evidence data for conditional sampling.
        :type evidenceData: pd.DataFrame, optional
        :param evidenceMapper: A mapping of evidence variables to their corresponding values.
        :type evidenceMapper: dict, optional
        :return: An instance of BayesianNetworkSynthesisWithEvidence.
        :rtype: BayesianNetworkSynthesisWithEvidence
        """
        
        data = pd.read_csv(path)
        evidenceData = pd.read_csv(evidenceData) if evidenceData is not None else None
        return BayesianNetworkSynthesisWithEvidence(components, data, reduction_factor, structure, structure_args, parameters, parameters_args, impossible, sampling_args, evidenceData, evidenceMapper)

    def __expand_dict(d):
        keys = d.keys()
        values = [v if isinstance(v, list) else [v] for v in d.values()]
        return [dict(zip(keys, combination))for combination in product(*values)]

    def _sample_iteration(self, row: pd.Series):
        """Sample from the Bayesian network for a specific row of evidence data.

        :param row: A row from the evidence DataFrame containing evidence values.
        :type row: pd.Series
        :return: A list of DataFrames containing the sampled data for the given evidence.
        :rtype: list of pd.DataFrame
        """

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
        """Sample from the Bayesian network, considering the provided evidence and filtering out impossible combinations. Each evidence row is processed in parallel to improve performance."""
        
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