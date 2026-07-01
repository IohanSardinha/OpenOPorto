
from ..StaticPopulationSynthesis import StaticPoplationSynthesis
from ...Component import ComponentSynthesis

class SingleComponentSynthesis(StaticPoplationSynthesis):
    def __init__(self, component_synthesis: ComponentSynthesis):
        self.component_synthesis = component_synthesis

    def run(self):
        return self.component_synthesis()

class ChainedSingleComponentSynthesis(StaticPoplationSynthesis):
    
    class Future():
        def __init__(self,att):
            self.att = att
        def __str__(self):
            return self.att
        def __int__(self):
            return self.att
    class FutureResult(Future):
        pass
    class FutureComponent(Future):
        pass

    def __init__(self, parts: dict[ComponentSynthesis:tuple]):
        self.parts = parts
        self.components = []
    
    def __parse_args(self, args, prevResult, prevComponent):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, self.FutureResult):
                if isinstance(prevResult, tuple) or isinstance(prevResult, list):
                    args[i] = prevResult[int(arg)]
                else:
                    args[i] = getattr(prevResult, str(arg))
            if isinstance(arg,self.FutureComponent):
                args[i] = getattr(prevComponent, str(arg))

        return tuple(args)
    
    def run(self):
        prevResult = None
        prevComponent = None
        for part, args in self.parts.items():
            prevComponent = part(*self.__parse_args(args, prevResult, prevComponent))
            self.components.append(prevComponent)
            synthesis = SingleComponentSynthesis(prevComponent)
            prevResult = synthesis.run()
        return prevResult