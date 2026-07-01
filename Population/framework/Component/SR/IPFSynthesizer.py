import numpy as np
import cloudpickle
import pandas as pd
from math import prod
import geopandas as gpd
from ...misc import cache
from ipfn.ipfn import ipfn
import multiprocessing as mp
from abc import ABC, abstractmethod
from .Integerizer import Integerizer
from itertools import combinations, product
from .SyntheticReconstruction import SyntheticReconstruction
from concurrent.futures import ProcessPoolExecutor
from ..ComponentSynthesis import ComponentSynthesis

IMPOSSIBILITIES_VAL = 1e-4

class IPF_Validator():

    @staticmethod
    def MAPE(pred, true, labels=None):
        raise NotImplementedError()

    @staticmethod #Mean Square Error
    def MSE(pred, true, labels=None):
        if labels is None:
            labels = [f"dim{i}" for i in range(pred.ndim)]

        mse_by_cat = {}
        for i, cat_name in enumerate(labels):
            other_axes = tuple(j for j in range(pred.ndim) if j != i)
            M_marginal = pred.sum(axis=other_axes)
            target_marginal = true[i]           
            diff = M_marginal - target_marginal
            mse_by_cat[cat_name] = np.sum(diff ** 2) / diff.size

        return mse_by_cat

    @staticmethod #Root Mean Square Error
    def RMSE(pred, true, labels=None):
        mse = IPF_Validator.MSE(pred, true, labels)
        for cat in mse:
            mse[cat] = np.sqrt(mse[cat])
        return mse
    
    @staticmethod #Freeman-Tukey-Read 
    def FTR(observed, expected, labels=None):
        if labels is None:
            labels = [f"dim{i}" for i in range(observed.ndim)]
        
        ftr_by_cat = {}
        for i, cat_name in enumerate(labels):
            other_axes = tuple(j for j in range(observed.ndim) if j != i)
            obs_marginal = observed.sum(axis=other_axes)
            exp_marginal = expected.sum(axis=other_axes)  # pre-integer IPF output
            ftr_by_cat[cat_name] = 4 * np.sum(
                (np.sqrt(obs_marginal) - np.sqrt(exp_marginal)) ** 2
            )
        
        return ftr_by_cat

    @staticmethod
    def combine_RMSE(rmse_a, size_a, rmse_b, size_b):
        sse_a = (rmse_a ** 2) * size_a
        sse_b = (rmse_b ** 2) * size_b
        return np.sqrt((sse_a + sse_b) / (size_a + size_b))

class IPFProcesser(ABC):

    def __init__(self, data, columns, impossibilities, asDF=False, labels=None, valueMapper={}, impossible_val = IMPOSSIBILITIES_VAL, correction_factor=1):
        self.data = data
        self.columns = columns
        self.impossibilities = impossibilities
        self.asDF = asDF
        self.labels = labels
        self.valueMapper = valueMapper
        self.impossible_val = impossible_val
        self.correction_factor = correction_factor

    def validate(self, data, marginals, labels):
        return {"RMSE": IPF_Validator.RMSE(data, marginals, labels),
                "FTR": IPF_Validator.FTR(data, self.Original_M, labels)}

    @abstractmethod
    def __call__(self):
        pass

class IPF2DProcess(IPFProcesser):

    def __call__(self):

        marginals = [self.data[dim].values for dim in self.columns]    
        self.marginals = marginals
        
        n_dims = len(self.columns)

        # Handle 2D case directly
        shape = tuple(len(sel) for sel in self.columns)
        M = np.ones(shape, dtype=float)
        
        # Apply impossible combinations
        for forb in self.impossibilities:
            indices = []
            valid = True
            for d in range(n_dims):
                try:
                    idx = self.columns[d].index(forb[d])
                except ValueError:
                    valid = False
                    break
                indices.append(idx)
            if valid:
                M[tuple(indices)] = self.impossible_val

        self.Original_M = M.copy()

        #M *= correction_factor

        # Run IPF with 1D marginals
        M = ipfn(M, marginals, [[0], [1]]).iteration()

        #M /= correction_factor

        self.pop = M

        return self.pop, self.validate(M, marginals, self.labels)

class IPFHighDimProcess(IPFProcesser):
    def __call__(self):
        marginals = []
        for dim in self.columns:
            marginals.append(self.data[dim].values)
        
        self.marginals = marginals

        n_dims = len(self.columns)
        combs = list(combinations(range(n_dims), n_dims - 1))
        next_marginals = []
        next_dimensions = []
        
        for comb in combs:
            current_dims = comb
            current_marginals = [marginals[i] for i in current_dims]
            shape = [len(m) for m in current_marginals]
            sub_M = np.ones(shape, dtype=float)
            
            # Apply impossible combinations relevant to current_dims
            
            allDims = [x for xs in self.columns for x in xs]
            temp_forb = [[imp_comb[d] for d in current_dims] for imp_comb in self.impossibilities]

            count = {}
            for f in temp_forb:
                x = "-".join([str(allDims.index(c)) for c in f])
                if x in count:
                    count[x][0] += 1
                else:
                    count[x] = [1, f]

            ohterDimsProd = prod([len(self.columns[i]) for i in range(len(self.columns)) if i not in comb])            
            
            # Fit (N-1)-dimensional marginal using 1D marginals
            sub_layers = [[i] for i in range(len(current_dims))]
            #sub_M *= correction_factor
            sub_M = ipfn(sub_M, current_marginals, sub_layers, max_iteration=1000).iteration()
            #sub_M /= correction_factor
            next_marginals.append(sub_M)
            next_dimensions.append(list(comb))  # Convert to list for ipfn compatibility
        
        # Initialize N-dimensional matrix
        shape = tuple(len(sel) for sel in self.columns)
        M = np.ones(shape, dtype=float)
        
        # Apply all N-dimensional impossible combinations
        for forb in self.impossibilities:
            indices = []
            for d in range(n_dims):
                idx = self.columns[d].index(forb[d])
                indices.append(idx)
            M[tuple(indices)] = self.impossible_val
        
        # Prepare layers for N-dimensional IPF
        layers = [list(comb) for comb in combs]

        # Run final IPF
        self.Original_M = M.copy()
        #M *= correction_factor
        M = ipfn(M, next_marginals, layers, max_iteration=1000000).iteration()
        #M /= correction_factor
        self.pop = M
        return M, self.validate(M, marginals, self.labels)

class IPFSynthesis(SyntheticReconstruction):
    
    def __init__(self, component: ComponentSynthesis.COMPONTENTS, integerizer:Integerizer, data, asDF=False, labels=None, valueMapper={}, correction_factor = 1):
        super().__init__(component)
        self.integerizer = integerizer
        self.data = data
        self.columns = integerizer.columns
        self.impossibilities = integerizer.impossibilities
        self.asDF = asDF
        self.labels = labels
        self.valueMapper = valueMapper
        self.correction_factor = correction_factor
    
    @staticmethod
    def fromGeoPackage(component: ComponentSynthesis.COMPONTENTS, integerizer:Integerizer, file_path:str, asDF=False, labels=None, valueMapper={}, correction_factor = 1):
        return IPFSynthesis(component, Integerizer, gpd.read_file(file_path), asDF=asDF, labels=labels, valueMapper=valueMapper, correction_factor=correction_factor)

    def synthesize(self, asDF=True):

        if len(self.columns) == 2:
            self.ipf = IPF2DProcess(self.data, self.columns, self.impossibilities, asDF=asDF, labels=self.labels, valueMapper=self.valueMapper, correction_factor=self.correction_factor)
        else:
            self.ipf = IPFHighDimProcess(self.data, self.columns, self.impossibilities, asDF=asDF, labels=self.labels, valueMapper=self.valueMapper, correction_factor=self.correction_factor)

        data, error = self.ipf()

        self.popErr = error

        integerData, int_err = self.integerizer(data)

        self.intErr = int_err

        self.pop = integerData
        
        return integerData if not (self.asDF and asDF) else self.array_to_dataframe(labels=self.labels, valueMapper=self.valueMapper), self.validate()

    def array_to_dataframe(self, labels=None, valueMapper={}):

        if labels is None:
            cols = [f"var{i+1}" for i in range(len(self.columns))]
        elif len(labels) != len(self.columns):
            raise "Labels and dimensions dont't match"
        else:
            cols = labels

        coords = list(product(*self.columns))
        values = self.pop.flatten()

        df = pd.DataFrame(coords, columns=cols)
        
        df = df.replace(valueMapper)

        df["value"] = values
        
        df = df[df["value"] > 0].reset_index(drop=True)

        return df

    def validate(self):
        
        return {
                "ipf":self.popErr,
                "integerization": self.intErr,
                "total":{
                         "rmse": IPF_Validator.RMSE(self.pop,self.ipf.marginals, self.labels),
                         "ftr" : IPF_Validator.FTR (self.pop,self.ipf.Original_M, self.labels)
                        }
                }

class IPFSynthesisWithSections(IPFSynthesis):
    def __init__(self, component: ComponentSynthesis.COMPONTENTS, integerizer:Integerizer, data, sectionVar, asDF=False, labels=None, valueMapper={}, correction_factor = 1):
        super().__init__(component, integerizer, data, asDF=asDF, labels=labels, valueMapper=valueMapper, correction_factor=correction_factor)
        self.sectionVar = sectionVar
    
    @staticmethod
    def fromGeoPackage(component: ComponentSynthesis.COMPONTENTS, integerizer:Integerizer, sectionVar, file_path, asDF=False, labels=None, valueMapper={}, correction_factor = 1):
        self = IPFSynthesisWithSections(component, integerizer, gpd.read_file(file_path), sectionVar, asDF=asDF, labels=labels, valueMapper=valueMapper, correction_factor=correction_factor)
        self.sectionShapes = self.data[[self.sectionVar,"geometry"]].rename(columns={self.sectionVar:"section"})
        return self
    
    @staticmethod
    def _multithread_synthesize_wrapper(pickled_method):
        synthesize = cloudpickle.loads(pickled_method)
        M, error = synthesize(False)
        return M, error

    def synthesize(self, asDF=True):
        ogData = self.data
        result = {}
        errors = {}

        max_workers = globals().get('__mt_cores__', mp.cpu_count())

        work_items = []
        for _, row in self.data.iterrows():
            worker = IPFSynthesis(self.component, self.integerizer, row, asDF=self.asDF, labels=self.labels, valueMapper=self.valueMapper, correction_factor=self.correction_factor)
            pickled_method = cloudpickle.dumps(worker.synthesize)
            work_items.append(pickled_method)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for future in [executor.submit(self._multithread_synthesize_wrapper, item) for item in work_items]:
                M, error = future.result()
                section_id = self.data.iloc[len(result)][self.sectionVar]
                result[section_id] = M
                errors[section_id] = error
        
        self.data =  ogData
        self.pop = result
        return result if not (self.asDF and asDF) else self.array_to_dataframe(labels=self.labels, valueMapper=self.valueMapper), errors
    
    def array_to_dataframe(self, labels=None, valueMapper={}):
        og = self.pop
        df = None
        started = False
        for sectionID,pop in og.items():
            self.pop = pop
            ndf = super().array_to_dataframe(labels, valueMapper)
            ndf.insert(0, "section",sectionID)
            if not started:
                df = ndf
                started = True
            else:
                df = pd.concat([df, ndf], ignore_index=False)
        self.pop = og
        df = df[df["value"] > 0].reset_index(drop=True)
        return df