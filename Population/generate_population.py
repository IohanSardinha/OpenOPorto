import shutil
import argparse
import importlib
from pathlib import Path

import geopandas as gpd

from framework.Static.Multiple.Merge import AttributeMatching
from framework.Component.SL.Probabilistic import BayesianNetworkSynthesisWithEvidence
from framework.Component.Sampling.SamplingSynthesis import ActivityChainSampler, Sampling
from framework.Component.ComponentSynthesis import ComponentSynthesis, COMPONTENTS
from oporto.IMOB.Processer import IMobProcesser
from framework.Static.Single.SingleComponentSynthesis import ChainedSingleComponentSynthesis
from framework.Component.Extras.LocationAssigner import HeuristicLocationAssigner
from framework.Component.Extras.Heuristic import PlaceCategoryMapper, IMobActivity
from framework.misc import BoundingBoxBuilder, cache
from external.MATSim import MATSimPopulationExporter

class OpenOportoPopulationGenerator(AttributeMatching):
    def __init__(self, config_path):
        self.load(config_path)

        self.bn = BayesianNetworkSynthesisWithEvidence\
                            .from_csv(COMPONTENTS.Attributes,
                                      self.config["FILES"]["IPUMS"],
                                      reduction_factor=self.config["REDUCTION_FACTOR"],
                                      evidenceData=self.config["FILES"]["BN_EVIDENCE"],
                                      evidenceMapper=self.config["EVIDENCE_MAPPER"]) 

        self.sectionsShapes = gpd.read_file(self.config["FILES"]["GEOPACKAGE"])[[self.config["SECTIONS_VAR"],"geometry"]].rename(columns={self.config["SECTIONS_VAR"]:"section"})

        imob = IMobProcesser().read(folder=self.config["FILES"]["IMOB_FOLDER"])

        ac_sampler = ActivityChainSampler(imob)
        
        super().__init__({COMPONTENTS.Attributes: self.bn,
                          COMPONTENTS.Activities: ac_sampler},
                          self.config["JOIN_COLS"], keyMapper={COMPONTENTS.Attributes:self.config["MATCH_MAPPER_BN"]}, 
                                                    prioritizeWhenMissing={COMPONTENTS.Attributes:self.config["PRIORITY_COLS"]})
    
    def run(self):
        super().run()

        location_assigner = HeuristicLocationAssigner(self.config["FILES"]["PLACES"],
                                             self.sectionsShapes, 
                                             PlaceCategoryMapper, 
                                             IMobActivity.HOME, 
                                             False)


        self.results = location_assigner.process(self.results,BoundingBoxBuilder().build(*self.config["BOUNDING_BOX"]))
        
        return self.results


    def load(self, path):
        path = Path(path).resolve()

        spec = importlib.util.spec_from_file_location("config_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self.config = module.config

    def export(self, path="."):
        self.results.to_csv(f"{path}/{self.config['FILES']['OUTPUT']}.csv")

        MATSimPopulationExporter(self.results).as_XML().export(f"{path}/{self.config['FILES']['OUTPUT']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Synthetic Population for OpenOPorto, based on the config file")
    parser.add_argument("config", help="Path to config file", nargs="?", default="config.py")
    parser.add_argument("--clear-cache", help="Clear cache before running", action="store_true")
    args = parser.parse_args()


    if args.clear_cache: shutil.rmtree("cache", ignore_errors=True)
    generator = OpenOportoPopulationGenerator(args.config)
    generator.results = generator.run()
    generator.export()
    print("Done")