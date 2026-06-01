from networkCreator.networkCreator import MATSimNetworkCreator, PT2MATSimWrapper, Logger
from networkCreator.scheduleMerger import merge_schedules
from networkCreator.vehicleMerger import merge_vehicles
import xml.etree.ElementTree as ET
from pathlib import Path
import importlib
import argparse

class OpenPortoNetworkGenerator:
    def __init__(self, config):
        self.config = config
    
    def find_vehicles(self, file):
        ns = {"m": "http://www.matsim.org/files/dtd"}
        tree = ET.parse(file)
        return set(map(lambda x: x.find("m:networkMode",ns).get("networkMode"), tree.findall("m:vehicleType", ns)))

    def generate(self):
        osmConfig = {
                    "keepPaths": "true",
                    "outputCoordinateSystem": self.config["CRS"],
                    }

        mapperConfig = {}

        creator_config = {
            "auto_install_requirements":self.config["AUTO_INSTALL_REQUIREMENTS"],
            "osm_config": osmConfig,
            "mapper_config": mapperConfig,
            "osm_url": self.config["OSM"]["URL"],
            "gtfs_crs": osmConfig["outputCoordinateSystem"],
            "skip_downloads": self.config["SKIP_DOWNLOADS"],
            "skip_cropping": self.config["SKIP_CROPPING"],
            "clean_tmp": self.config["CLEAN_TMP"],
            "osm_download_path": self.config["OSM"]["FILE"],
            "osm_crop_path": self.config["OSM"]["CROP_FILE"],
        }

        if "BOUNDING_BOX" in self.config["OSM"]:
            creator_config["osm_crop_bbox"] = self.config["OSM"]["BOUNDING_BOX"]
        elif "CROP_RELATION" in self.config["OSM"]:
            creator_config["osm_crop_relation"] = self.config["OSM"]["CROP_RELATION"]

        last_schedules = None
        last_vehicles = None
        last_network_path = None
        modesToKeepOnCleanUp = set()
        for i, (name, pt) in enumerate(self.config["PUBLIC_TRANSPORT"].items()):
            
            creator_config["gtfs_url"] = pt["URL"] if "URL" in pt else None
            creator_config["gtfs_date"] = pt["DATE"]
            creator_config["gtfs_download_path"] = f".tmp/gtfs_{name.lower()}.zip"
            creator_config["gtfs_path"] = creator_config["gtfs_download_path"]
            creator_config["output_network_path"] = f".tmp/{name.lower()}_network.xml" if i != len(self.config["PUBLIC_TRANSPORT"])-1 \
                                                                             else self.config["OUTPUT_NETWORK"]
            creator_config["unmapped_schedule_path"] = f".tmp/{name.lower()}_unmapped_schedule.xml"
            creator_config["vehicles_path"] = f".tmp/{name.lower()}_vehicles.xml"
            creator_config["skip_new_network"] = (i != 0)
            creator_config["mapper_config"]["outputNetworkFile"] = creator_config["output_network_path"]
            creator_config["mapper_config"]["config_path"] = f".tmp/{name.lower()}_mapper_config.xml"
            creator_config["mapper_config"]["outputScheduleFile"] = f".tmp/{name.lower()}_schedule.xml"
            creator_config["mapper_config"]["outputStreetNetworkFile"] = f".tmp/{name.lower()}_output_street_network.xml"
            
            restore_downloads_config = None
            if "LOCAL_PATH" in pt and not "URL" in pt:
                creator_config["gtfs_download_path"] = pt["LOCAL_PATH"]
                restore_downloads_config = creator_config["skip_downloads"]
                creator_config["skip_downloads"] = True
                

            if len(modesToKeepOnCleanUp) > 0:
                creator_config["mapper_config"]["modesToKeepOnCleanUp"] = ",".join(modesToKeepOnCleanUp|{"car"})

            if i > 0: 
                creator_config["mapper_config"]["inputNetworkFile"] = last_network_path
            
            nc = MATSimNetworkCreator(creator_config, PT2MATSimWrapper, log_level=Logger.Level.INFO)
            nc.create_network([creator_config])

            if i == len(self.config["PUBLIC_TRANSPORT"])-1:
                output_schedule = self.config["OUTPUT_SCHEDULE"]
                output_vehicles = self.config["OUTPUT_VEHICLES"]
            else:
                output_schedule = ".tmp/"+"_".join(list(self.config["PUBLIC_TRANSPORT"].keys())[:i+1])+"joint_schedule.xml"
                output_vehicles = ".tmp/"+"_".join(list(self.config["PUBLIC_TRANSPORT"].keys())[:i+1])+"joint_vehicles.xml"

            if i > 0:
                
                curr_name = list(self.config["PUBLIC_TRANSPORT"].keys())[i-1],list(self.config["PUBLIC_TRANSPORT"].keys())[i]
                if i == 1:
                    last_name = list(self.config["PUBLIC_TRANSPORT"].keys())[i-1],list(self.config["PUBLIC_TRANSPORT"].keys())[i-1]
                else:
                    last_name = ""

                nc.logger.info(f"Merging schedules: {creator_config['mapper_config']['outputScheduleFile']} + {last_schedules} into {output_schedule}")
                merge_schedules(creator_config["mapper_config"]["outputScheduleFile"], last_schedules, output_schedule, last_name, curr_name)
                
                nc.logger.info(f"Merging vehicles: {creator_config['vehicles_path']} + {last_vehicles} into {output_vehicles}", last_name, curr_name)
                merge_vehicles(creator_config["vehicles_path"], last_vehicles, output_vehicles)
                last_schedules = output_schedule
                last_vehicles = output_vehicles
            else:
                last_schedules = creator_config["mapper_config"]["outputScheduleFile"]
                last_vehicles = creator_config["vehicles_path"]

            modesToKeepOnCleanUp |= self.find_vehicles(last_vehicles)

            last_network_path = creator_config["output_network_path"]
            
            if restore_downloads_config is not None:
                creator_config["skip_downloads"] = restore_downloads_config

def load_config(path):
    path = Path(path).resolve()

    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.config

def main():
    parser = argparse.ArgumentParser(description="Setup the physical network files for OpenOPorto, based on the config file")
    parser.add_argument("config", help="Path to config file", nargs="?", default="config.py")
    args = parser.parse_args()

    config = load_config(args.config)
        
    OpenPortoNetworkGenerator(config).generate()

if __name__ == "__main__":
    main()