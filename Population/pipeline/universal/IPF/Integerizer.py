import numpy as np
from ...ProcessStep import ProcessStep

class DefaultIntegerizer(ProcessStep):
    
    def __init__(self, columns=[], impossibilities=[]):
        self.columns = columns
        self.impossibilities = impossibilities

    def __setImpossiblesAsZeros(self, data):
        
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

    def process(self, data):

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
    
    def validate(self, data):
        
        diff = data - self.continuous
        rmse = np.sqrt(np.sum(diff ** 2) / diff.size)
        
        
        ftr = 4 * np.sum(
            (np.sqrt(data) - np.sqrt(self.continuous)) ** 2
        )
        
        return {
            "rmse": rmse,
            "ftr": ftr
        }
    
class MonteCarloSamplerIntegerizer(ProcessStep):
    
    def __init__(self):
        raise NotImplementedError("MonteCarloSamplerIntegerizer is not implemented yet.")

    def process(self, data):
        raise NotImplementedError("MonteCarloSamplerIntegerizer is not implemented yet.")
    
    def validate(self, data):
        raise NotImplementedError("MonteCarloSamplerIntegerizer is not implemented yet.")