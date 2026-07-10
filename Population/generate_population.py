import shutil
import argparse
import importlib
from pathlib import Path

from framework.Static.Multiple.Merge import AttributeMatching
from framework.Component.SR.IPFSynthesizer import IPFSynthesisWithSections
from framework.Component.SR.Integerizer import DefaultIntegerizer
from framework.Component.Sampling.SamplingSynthesis import ActivityChainSampler, Sampling
from framework.Component.ComponentSynthesis import ComponentSynthesis
from oporto.IMOB.Processer import IMobProcesser
from framework.Static.Single.SingleComponentSynthesis import ChainedSingleComponentSynthesis
from framework.Component.Extras.LocationAssigner import HeuristicLocationAssigner
from framework.Component.Extras.Heuristic import PlaceCategoryMapper, IMobActivity
from framework.misc import BoundingBoxBuilder, cache
from external.MATSim import MATSimPopulationExporter

class OpenOportoPopulationGenerator(AttributeMatching):
    def __init__(self, config_path):
        self.load(config_path)

        integerizer_H = DefaultIntegerizer(self.config["DIMENSIONS"]("H"), self.config["IMPOSSIBILITIES"]("H"))

        sample_size = 0.005
        
        ipf_args = (
            ComponentSynthesis.COMPONTENTS.Attributes,
            integerizer_H,
            self.config["SECTIONS_VAR"],
            self.config["FILES"]["GEOPACKAGE"],
            True,  # asDF
            self.config["COLS"],  # labels
            self.config["DIM_VALUE_MAP"]("H"),  # valueMapper
            self.config["CORRECTION_FACTOR"],  # correction_factor
        )

        sample_args = (
            ComponentSynthesis.COMPONTENTS.Attributes,
            ChainedSingleComponentSynthesis.FutureResult(0),
            sample_size
        )

        self.ipf = ChainedSingleComponentSynthesis({IPFSynthesisWithSections.fromGeoPackage: ipf_args, Sampling: sample_args})

        imob = IMobProcesser().read(folder=self.config["FILES"]["IMOB_FOLDER"])

        ac_sampler = ActivityChainSampler(imob)
        
        super().__init__({ComponentSynthesis.COMPONTENTS.Attributes: self.ipf.run,
                          ComponentSynthesis.COMPONTENTS.Activities: ac_sampler},
                          self.config["JOIN_COLS"], keyMapper={ComponentSynthesis.COMPONTENTS.Attributes:self.config["MATCH_MAPPER"]}, 
                                                    prioritizeWhenMissing={ComponentSynthesis.COMPONTENTS.Attributes:self.config["PRIORITY_COLS"]})
    
    def run(self):
        super().run()

        location_assigner = HeuristicLocationAssigner(self.config["FILES"]["PLACES"],
                                             self.ipf.components[0].sectionShapes, 
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
        self.results.to_csv(f"{path}/synthetic_population.csv")

        MATSimPopulationExporter(self.results).as_XML().export(f"{path}/population.xml")


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