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
from framework.Component.SR.IPFSynthesizer import IPFSynthesisWithSections
from framework.Component.SR.Integerizer import DefaultIntegerizer

class OpenOportoPopulationGenerator(AttributeMatching):
    def __init__(self, config_path, **kwargs):
        self.load(config_path)

        attribute_synthesis_method = self.config.get("ATTRIBUTE_SYNTHESIS_METHOD", kwargs.get("attribute_synthesis_method", "BN"))

        if attribute_synthesis_method == "BN" or attribute_synthesis_method is None:
            self.setup_bn()
        elif attribute_synthesis_method == "IPF":
            self.setup_ipf()
        else:
            raise ValueError(f"Unknown attribute synthesis method: {attribute_synthesis_method}")

        self.sectionsShapes = gpd.read_file(self.config["FILES"]["GEOPACKAGE"])[[self.config["SECTIONS_VAR"],"geometry"]].rename(columns={self.config["SECTIONS_VAR"]:"section"})

        imob = IMobProcesser().read(folder=self.config["FILES"]["IMOB_FOLDER"])

        ac_sampler = ActivityChainSampler(imob)
        
        super().__init__({COMPONTENTS.Attributes: self.attributeSynthesis,
                          COMPONTENTS.Activities: ac_sampler},
                          self.config["JOIN_COLS"], keyMapper={COMPONTENTS.Attributes:self.matchMapper}, 
                                                    prioritizeWhenMissing={COMPONTENTS.Attributes:self.config["PRIORITY_COLS"]})

    def setup_bn(self):
        self.bn = BayesianNetworkSynthesisWithEvidence\
                                    .from_csv(COMPONTENTS.Attributes,
                                              self.config["FILES"]["IPUMS"],
                                              reduction_factor=self.config["REDUCTION_FACTOR"],
                                              evidenceData=self.config["FILES"]["BN_EVIDENCE"],
                                              evidenceMapper=self.config["EVIDENCE_MAPPER"]) 
        
        self.attributeSynthesis = self.bn

        self.matchMapper = self.config["MATCH_MAPPER_BN"]

    def setup_ipf(self):
        integerizer_H = DefaultIntegerizer(self.config["DIMENSIONS"], self.config["IMPOSSIBILITIES"])
 
        ipf_args = (
            ComponentSynthesis.COMPONTENTS.Attributes,
            integerizer_H,
            self.config["SECTIONS_VAR"],
            self.config["FILES"]["GEOPACKAGE"],
            True,  # asDF
            self.config["COLS"],  # labels
            self.config["DIM_VALUE_MAP"],  # valueMapper
            self.config["CORRECTION_FACTOR"],  # correction_factor
        )

        sample_args = (
            ComponentSynthesis.COMPONTENTS.Attributes,
            ChainedSingleComponentSynthesis.FutureResult(0),
            self.config["REDUCTION_FACTOR"]
        )

        self.ipf = ChainedSingleComponentSynthesis({IPFSynthesisWithSections.fromGeoPackage: ipf_args, Sampling: sample_args})

        self.attributeSynthesis = self.ipf.run

        self.matchMapper = self.config["MATCH_MAPPER_IPF"]

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
    parser.add_argument("--attribute-synthesis-method", help="Attribute synthesis method to use (BN or IPF for now)", choices=["BN", "IPF"], default=None)

    args = parser.parse_args()


    if args.clear_cache: shutil.rmtree("cache", ignore_errors=True)
    generator = OpenOportoPopulationGenerator(args.config, attribute_synthesis_method=args.attribute_synthesis_method)
    generator.results = generator.run()
    generator.export()
    print("Done")