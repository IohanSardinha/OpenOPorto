import pickle
import argparse
import importlib
import pandas as pd
from pathlib import Path
from pipeline.oporto.IMob.Processer import IMobProcesser
from pipeline.oporto.misc import build_id
from pipeline.external.MATSim import MATSimPopulationExporter
from pipeline.universal.IPF.Integerizer import DefaultIntegerizer
from pipeline.oporto.data.HeuristicMatcher import PlaceCategoryMapper
from pipeline.universal.misc import BoundingBoxBuilder, PlacesGenericFormat, JOIN_MODE
from pipeline.universal.ActivityChain.locationAssigner import HeuristicLocationAssigner
from pipeline.universal.ActivityChain.defaultActivityMatcher import DefaultActivityMatcher
from pipeline.universal.IPF.ipfPopulationSynthesizer import IPFPopulationSynthesisWithSections
from pipeline.pipeline import MultiStepPopulationSynthesis, PostLocationAssignActivityChainMatcher
from pipeline.oporto.IMob.ActivityTypes import IMobActivity

class OpenOportoPopulationGenerator(MultiStepPopulationSynthesis):
    def __init__(self, config):
        self.config = config

    def load_cache(self, doc):
        if Path(f"cache/{doc}.pkl").exists() and self.config.get("CACHE", False):
            print(f"Loading {doc} from cache.")
            with open(f"cache/{doc}.pkl", "rb") as f:
                return pickle.load(f)
        return None  

    def save_cache(self, doc, data):
        if self.config.get("CACHE", False):
            with open(f"cache/{doc}.pkl", "wb") as f:
                pickle.dump(data, f)

    def generate_population(self):

        if self.config.get("CACHE", False):
            Path("cache").mkdir(exist_ok=True)

        self.persons = IMobProcesser.read(self.config["FILES"]["HOUSEHOLDS"], self.config["FILES"]["EXPENSES"], self.config["FILES"]["VEHICLES"], self.config["FILES"]["INCOMES"], self.config["FILES"]["INDIVIDUALS"], self.config["FILES"]["PASSES"], self.config["FILES"]["TRIPS"])

        self.boundingBox = BoundingBoxBuilder().build(*self.config["BOUNDING_BOX"])

        self.places = PlacesGenericFormat(self.config["FILES"]["PLACES"])
    
        self.ipfMen = IPFPopulationSynthesisWithSections(DefaultIntegerizer(self.config["DIMENSIONS"]("H"), self.config["IMPOSSIBILITIES"]("H")), self.config["SECTIONS_VAR"], asDF=True, labels=self.config["COLS"], valueMapper=self.config["DIM_VALUE_MAP"]("H"), correction_fac=self.config["CORRECTION_FACTOR"])\
                                                .fromGeoPackage(self.config["FILES"]["GEOPACKAGE"])

        self.ipfWomen = IPFPopulationSynthesisWithSections(DefaultIntegerizer(self.config["DIMENSIONS"]("M"), self.config["IMPOSSIBILITIES"]("M")), self.config["SECTIONS_VAR"], asDF=True, labels=self.config["COLS"], valueMapper=self.config["DIM_VALUE_MAP"]("M"), correction_fac=self.config["CORRECTION_FACTOR"])\
                                                .fromGeoPackage(self.config["FILES"]["GEOPACKAGE"])

        assigner = HeuristicLocationAssigner(self.places, self.ipfMen.sectionShapes, PlaceCategoryMapper,IMobActivity.HOME, silent=self.config["SILENT"], print_with_display=self.config["PRINT_WITH_DISPLAY"])
        self.ActivityChainMatcher = PostLocationAssignActivityChainMatcher(DefaultActivityMatcher(), assigner)

        print(f"Generating OpenOporto Synthetic Population...")

        self.process()

        if "JSON" in self.config["FILES"]:
            print(f"Exporting synthetic population as JSON to {self.config['FILES']['JSON']}...")
            self.export(self.config["FILES"]["JSON"])

        MATSimPopulationExporter(self.matched_population, id_builder=build_id).as_XML().export(self.config["FILES"]["OUTPUT"])
        print(f"Pipeline test population successfully exported to {self.config['FILES']['OUTPUT']}!")

    def process(self):

        cached_population = self.load_cache("synthesized_population")
        cache_error = self.load_cache("synthesis_error")

        if cached_population is None:
            
            print(self.ipfMen.data)
            self.PopulationSynthesizer = self.ipfMen
            self.synthesize((self.config["DIMENSIONS"]("H"), self.config["IMPOSSIBILITIES"]("H")))
            menDf = self.synthesized_population
            print(menDf)
            menErr = self.synthesis_error
            menDf["gender"] = "Masculino"

            self.PopulationSynthesizer = self.ipfWomen
            self.synthesize((self.config["DIMENSIONS"]("M"), self.config["IMPOSSIBILITIES"]("M")))
            womenDf = self.synthesized_population
            print(womenDf)
            womenErr = self.synthesis_error
            womenDf["gender"] = "Feminino"

            self.synthesized_population = pd.concat([menDf, womenDf], ignore_index=False)

            self.synthesized_population = self.synthesized_population[self.synthesized_population["residence"] == "Live in Portugal"]

            self.synthesis_error = {"H": menErr, "M": womenErr}

            self.synthesized_population.to_csv("cache/synthesized_population.csv", index=False)
            self.save_cache("synthesized_population", self.synthesized_population)
            self.save_cache("synthesis_error", self.synthesis_error)

        else:
            self.synthesized_population = cached_population
            self.synthesis_error = cache_error

        self.match(((self.persons,
                    (self.synthesized_population, self.persons, self.config["JOIN_COLS"], self.config["MATCH_MAPPER"], JOIN_MODE.BOTH, self.config["REDUCTION_FACTOR"], self.config["PRIORITY_COLS"]),
                    (self.persons, self.boundingBox))))

        return self.matched_population, self.validate()

def load_config(path):
    path = Path(path).resolve()

    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.config

def main():
    parser = argparse.ArgumentParser(description="Generate a Synthetic Population for OpenOPorto, based on the config file")
    parser.add_argument("config", help="Path to config file", nargs="?", default="config.py")
    args = parser.parse_args()

    config = load_config(args.config)
        
    OpenOportoPopulationGenerator(config).generate_population()

if __name__ == "__main__":
    main()
