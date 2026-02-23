import numpy as np
import pandas as pd
from math import prod
import geopandas as gpd
from ipfn.ipfn import ipfn
from ...ProcessStep import ProcessStep
from itertools import combinations, product

IMPOSSIBILITIES_VAL = 0.0001

class IPF2DProcess(ProcessStep):
    def process(self, data, columns, impossibilities, asDF=False, labels=None, valueMapper={}, impossible_val = IMPOSSIBILITIES_VAL, correction_fac = 1):

        marginals = []
        for dim in columns:
            marginals.append(data[dim].values)
        
        n_dims = len(columns)

        # Handle 2D case directly
        shape = tuple(len(sel) for sel in columns)
        M = np.ones(shape, dtype=float)
        
        # Apply impossible combinations
        for forb in impossibilities:
            indices = []
            valid = True
            for d in range(n_dims):
                try:
                    idx = columns[d].index(forb[d])
                except ValueError:
                    valid = False
                    break
                indices.append(idx)
            if valid:
                M[tuple(indices)] = impossible_val
        
        # Run IPF with 1D marginals
        M = ipfn(M, marginals, [[0], [1]]).iteration()

        return M if not asDF else self.array_to_dataframe(labels=self.labels, valueMapper=self.valueMapper), self.validate(M) 

    def validate(self, data):
        return data

class IPFHighDimProcess(ProcessStep):
    def process(self, data, columns, impossibilities, asDF=False, labels=None, valueMapper={}, impossible_val = IMPOSSIBILITIES_VAL, correction_fac = 1):
        marginals = []
        for dim in columns:
            marginals.append(data[dim].values)
        
        n_dims = len(columns)
        combs = list(combinations(range(n_dims), n_dims - 1))
        next_marginals = []
        next_dimensions = []
        
        for comb in combs:
            current_dims = comb
            current_marginals = [marginals[i] for i in current_dims]
            shape = [len(m) for m in current_marginals]
            sub_M = np.ones(shape, dtype=float)
            
            # Apply impossible combinations relevant to current_dims
            
            allDims = [x for xs in columns for x in xs]
            temp_forb = [[imp_comb[d] for d in current_dims] for imp_comb in impossibilities]

            count = {}
            for f in temp_forb:
                x = "-".join([str(allDims.index(c)) for c in f])
                if x in count:
                    count[x][0] += 1
                else:
                    count[x] = [1, f]

            ohterDimsProd = prod([len(columns[i]) for i in range(len(columns)) if i not in comb])            
            
            # Fit (N-1)-dimensional marginal using 1D marginals
            sub_layers = [[i] for i in range(len(current_dims))]
            sub_M = ipfn(sub_M, current_marginals, sub_layers, max_iteration=1000).iteration()
            
            next_marginals.append(sub_M)
            next_dimensions.append(list(comb))  # Convert to list for ipfn compatibility
        
        # Initialize N-dimensional matrix
        shape = tuple(len(sel) for sel in columns)
        M = np.ones(shape, dtype=float)
        
        # Apply all N-dimensional impossible combinations
        for forb in impossibilities:
            indices = []
            for d in range(n_dims):
                idx = columns[d].index(forb[d])
                indices.append(idx)
            M[tuple(indices)] = impossible_val
        
        # Prepare layers for N-dimensional IPF
        layers = [list(comb) for comb in combs]

        # Run final IPF
        M = ipfn(M, next_marginals, layers, max_iteration=1000000).iteration()

        return M if not asDF else self.array_to_dataframe(labels=self.labels, valueMapper=self.valueMapper), self.validate(M)

    def validate(self, data):
        return data

class IPFPopulationSynthesis(ProcessStep):
    
    def __init__(self, Integerizer:ProcessStep, asDF=False, labels=None, valueMapper={}):
        self.integerizer = Integerizer
        self.asDF = asDF
        self.labels = labels
        self.valueMapper = valueMapper

    def fromGeoPackage(self, file_path:str):
        self.data = gpd.read_file(file_path)
        return self

    def process(self, columns, impossibilities, asDF=True):
        
        self.columns = columns

        if len(columns) == 2:
            data, error = IPF2DProcess().process(self.data, columns, impossibilities, asDF=asDF, labels=self.labels, valueMapper=self.valueMapper)
        else:
            data, error = IPFHighDimProcess().process(self.data, columns, impossibilities, asDF=asDF, labels=self.labels, valueMapper=self.valueMapper)

        integerData, int_err = self.integerizer.process(data)

        self.pop = integerData
        
        return integerData if not (self.asDF and asDF) else self.array_to_dataframe(labels=self.labels, valueMapper=self.valueMapper), error

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

    def validate(self, data):
        pass

class IPFPopulationSynthesisWithSections(IPFPopulationSynthesis):
    def __init__(self, Integerizer, sectionVar, asDF=False, labels=None, valueMapper={}):
        super().__init__(Integerizer, asDF=asDF, labels=labels, valueMapper=valueMapper)
        self.sectionVar = sectionVar

    def process(self, columns, impossibilities, asDF=True):
        ogData = self.data
        result = {}
        errors = {}
        for _, row in self.data.iterrows():
            self.data = row
            M, error = super().process(columns, impossibilities, asDF=False)
            result[row[self.sectionVar]] = M
            errors[row[self.sectionVar]] = M
        self.data =  ogData
        self.pop = result
        return result if not (self.asDF and asDF) else self.array_to_dataframe(labels=self.labels, valueMapper=self.valueMapper), errors
    
    def fromGeoPackage(self, file_path):
        super().fromGeoPackage(file_path)
        self.sectionShapes = self.data[[self.sectionVar,"geometry"]].rename(columns={self.sectionVar:"section"})
        return self

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