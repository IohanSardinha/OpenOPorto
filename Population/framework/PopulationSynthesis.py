from abc import ABC, abstractmethod

class PopulationSynthesis(ABC):
        
    def load(self):
        pass
    
    @abstractmethod
    def run(self):
        pass

    def export(self):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}"
