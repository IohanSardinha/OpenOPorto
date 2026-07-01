from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

class ComponentSynthesis(ABC):
    COMPONTENTS = Enum("COMPONTENTS", ("Attributes", "Relations", "Activities"))

    def __init__(self, component: COMPONTENTS):
        self.component = component

    def before(self):
        pass
    
    def after(self, result)->Tuple[bool, any]|None:
        pass

    @abstractmethod
    def synthesize(self):
        pass

    def __parse_after(self, result, after_result):
        if after_result is not None:
            if isinstance(after_result, tuple) and len(after_result) == 2 and after_result[0] is True:
                return after_result[1]
        return result

    def __call__(self):
        self.before()
        result = self.synthesize()
        after_result = self.after(result)
        return self.__parse_after(result, after_result)

    def __str__(self):
        return f"{self.__class__.__name__}|{self.component.name}"


class CombinedCapable(ComponentSynthesis, ABC):
    def __init__(self, components: set[ComponentSynthesis.COMPONTENTS]):
        self.components = components