from .ProcessStep import ProcessStep
from enum import Enum
import json

class PostLocationAssignActivityChainMatcher(ProcessStep):
    def __init__(self, Matcher, Assigner):
        self.matcher = Matcher
        self.assigner = Assigner

    def process(self, trips, MatcherArgs, AssignerArgs):
        matchedPopulation, err = self.matcher.process(*MatcherArgs)

        locations, errors = self.assigner.process(matchedPopulation, *AssignerArgs)
        
        records = matchedPopulation.to_dict("records")

        combined = [
            {
                "attributes": {k: v for k, v in person.items() if k != "match"},
                "trips": [
                    {**leg, "x": loc.x, "y": loc.y}
                    for leg, loc in zip(trips[person["match"]]["legs"], locations[person["match"]])
                ],
                "tripDesc": trips[person["match"]]["tripDesc"],
            }
            for i, person in enumerate(records)
        ]

        return combined, self.validate(err, errors)

    def validate(self, matcherError, assignerErrors):
        return {"matcher": matcherError, "assigner": assignerErrors}   

class MultiStepPopulationSynthesis(ProcessStep):
    class ItermidiateResult(Enum):
        SYNTHESIZED_POPULATION = 1
        SYNTHESIZED_ERROR = 2

    def export(self, path, what=None):
        if what == self.ItermidiateResult.SYNTHESIZED_POPULATION:
            what = self.synthesized_population
        else:
            what = self.matched_population

        with open(path, "w") as f:
            json.dump(what, f, indent=4, default=str)

    def __init__(self, PopulationSynthesizer, ActivityChainMatcher):
        self.PopulationSynthesizer = PopulationSynthesizer
        self.ActivityChainMatcher = ActivityChainMatcher

    def __synthesize(self, synthesizerArgs):
        population, error = self.PopulationSynthesizer.process(*synthesizerArgs)
        self.synthesized_population = population
        self.synthesis_error = error
    
    def synthesize(self, synthesizerArgs):
        self.__synthesize(synthesizerArgs)
        return self
    
    def __map_args(self, arg):
        if type(arg) is self.ItermidiateResult:
            if arg == self.ItermidiateResult.SYNTHESIZED_POPULATION:
                return self.synthesized_population
            elif arg == self.ItermidiateResult.SYNTHESIZED_ERROR:
                return self.synthesis_error
        else:
            return arg

    def replace(self, args):
        return tuple(self.replace(arg) if isinstance(arg, tuple) else self.__map_args(arg) for arg in args)

    def __match(self, matcherArgs):
        if self.synthesized_population is None:
            raise Exception("Population must be synthesized before matching.")

        matcherArgs = self.replace(matcherArgs)

        population, error = self.ActivityChainMatcher.process(*matcherArgs)
        self.matched_population = population
        self.matching_error = error
    
    def match(self, matcherArgs):
        self.__match(matcherArgs)
        return self

    def process(self, synthesizerArgs, matcherArgs):
        self.__synthesize(synthesizerArgs)
        self.__match(matcherArgs)
        return self.matched_population, self.validate()
    
    def validate(self):
        return {"synthesis": self.synthesis_error, "matching": self.matching_error}