import shutil
from enum import Enum
import requests
from pathlib import Path
from subprocess import run, PIPE, STDOUT, DEVNULL, CalledProcessError
from abc import ABC, abstractmethod
from .defaults.pt2matsim_default_config import DEFAULT_CONFIG as PT2MATSim_DEFAULT_CONFIG
from .defaults.pt2matsim_default_mapper_config import DEFAULT_MAPPER_CONFIG as PT2MATSim_DEFAULT_MAPPER_CONFIG

def __my_run__(self, cmd):
    self.logger.debug("Running:", " ".join(cmd))
    if self.logger.level.value >= Logger.Level.DEBUG.value:
        #result = 
        run(cmd, check=True)#, stdout=STDOUT, stderr=STDOUT)#, text=True)
        #for line in result.stdout.splitlines():
        #    self.logger.subprocess(line)

    else:
        run(cmd, check=True, stdout=DEVNULL, stderr=DEVNULL)

class RequirementError(RuntimeError):
    pass

class Logger:
    class Level(Enum):
        SILENT = 0
        INFO = 1
        DEBUG = 2
        SUBPROCESS = 3

    def __init__(self, level=Level.SILENT):
        self.level = level
    
    def info(self, *args, **kwargs):
        if self.level.value >= Logger.Level.INFO.value:
            print("[INFO]", *args, **kwargs)

    def debug(self, *args, **kwargs):
        if self.level.value >= Logger.Level.DEBUG.value:
            print("[DEBUG]", *args, **kwargs)

    def subprocess(self, *args, **kwargs):
        if self.level.value >= Logger.Level.SUBPROCESS.value:
            print("[SUBPROCESS]", *args, **kwargs)

class NetworkCreatorEngine(ABC):
    def __init__(self, logger: Logger = None):
        self.logger = logger if logger != None else Logger(Logger.Level.SILENT)

    @abstractmethod
    def createNetwork(self, osm_path, output_network_path):
        pass

class PT2MATSimWrapper(NetworkCreatorEngine):
    def __init__(self, logger=None, version="24.4"):
        super().__init__(logger)

        self._run = lambda cmd: __my_run__(self,cmd)

        self.check_requirements()

        if not Path(".tmp/pt2matsim.jar").exists():
            
            self.logger.info("Downloading pt2matsim JAR...")
            self.logger.debug(f"Using pt2matsim version {version}")

            Path(".tmp").mkdir(parents=True, exist_ok=True)
            url = f"https://repo.matsim.org/repository/matsim/org/matsim/pt2matsim/{version}/pt2matsim-{version}-shaded.jar"
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(".tmp/pt2matsim.jar", "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
    
    def check_requirements(self):
        if shutil.which("java") is None:
            if self.config.get("auto_install_requirements", False):
                self.logger.info("Missing java, will install it")
                self._run(["sudo", "apt", "install", "default-jre"])
            else:
                raise RequirementError("Missing java")

    def new_config_file(self, config, output_path):
        self.logger.info(f"Writing OSM config to {output_path}")
        with open(output_path, "w") as f:
            f.write(PT2MATSim_DEFAULT_CONFIG(config))

    def new_mapper_config_file(self, mapper_config, output_path):
        self.logger.info(f"Writing mapper config to {output_path}")
        with open(output_path, "w") as f:
            f.write(PT2MATSim_DEFAULT_MAPPER_CONFIG(mapper_config))

    def create_unmapped_network(self, config_path):
        self._run(["java", "-cp", ".tmp/pt2matsim.jar:libs", "org.matsim.pt2matsim.run.Osm2MultimodalNetwork", config_path])
    
    def create_unmapped_schedule(self, GTFS_path, date, crs, schedule_output_path, vehicles_output_path):
        self._run(["java", "-cp", ".tmp/pt2matsim.jar:libs", "org.matsim.pt2matsim.run.Gtfs2TransitSchedule", GTFS_path, date, crs, schedule_output_path, vehicles_output_path])    
    
    def map_schedule(self, mapper_config_path):
        self._run(["java", "-cp", ".tmp/pt2matsim.jar", "org.matsim.pt2matsim.run.PublicTransitMapper", mapper_config_path])

    def createNetwork(self, osm_path, output_network_path, config):

        osmConfig = config.get("osm_config", {})
        mapperConfig = config.get("mapper_config", {})
        gtfs_path = config.get("gtfs_path", None)
        getfs_date = config.get("gtfs_date", None)
        crs = config.get("gtfs_crs", None)

        has_pt = (gtfs_path is not None) and (getfs_date is not None) and (crs is not None)

        self.logger.info(f"Public transport?: {has_pt}")

        osmConfig["osmFile"] = osmConfig.get("osmFile", osm_path)
        mapperConfig["outputNetworkFile"] = mapperConfig.get("outputNetworkFile", output_network_path)

        # Create Default OSM Config file
        if has_pt:
            osmConfig["outputNetworkFile"] = osmConfig.get("baseNetworkFile", ".tmp/base_network.xml")
            mapperConfig["outputNetworkFile"] = mapperConfig.get("outputNetworkFile", output_network_path)
        else:
            osmConfig["outputNetworkFile"] = osmConfig.get("outputNetworkFile", output_network_path)

        # Create the unmmapped network
        if config.get("skip_new_network", False):
            self.logger.info(f"Skipping generating unmapped osm network : {osmConfig['outputNetworkFile']}")
        else:
            osm_config_path = osmConfig.get("config_path", f".tmp/pt2matsim_osm_config.xml")
            self.new_config_file(osmConfig, osm_config_path)
            self.logger.info(f"Generating unmapped osm network: {osmConfig['outputNetworkFile']}")
            self.create_unmapped_network(osm_config_path)

        if gtfs_path == None or getfs_date == None or crs == None:
            return

        unmapped_schedule_path = config.get("unmapped_schedule_path", ".tmp/unmapped_schedule.xml")
        unmapped_vehicles_path = config.get("vehicles_path", "vehicles.xml")

        mapperConfig["inputScheduleFile"] = unmapped_schedule_path
        mapperConfig["inputNetworkFile"] = mapperConfig.get("inputNetworkFile", osmConfig["outputNetworkFile"])

        # Create Default Mapper Config
        mapper_config_path = mapperConfig.get("config_path", f".tmp/pt2matsim_mapper_config.xml")
        self.new_mapper_config_file(mapperConfig, mapper_config_path)

        # Create the unmapped schedule
        self.logger.info(f"Generating unmapped schedule for {gtfs_path}: {unmapped_schedule_path}, {unmapped_vehicles_path}")
        self.create_unmapped_schedule(gtfs_path, getfs_date, crs, unmapped_schedule_path, unmapped_vehicles_path)
        
        # Map the metro schedule to the network
        self.logger.info(f"Mapping schedule to final network: {mapperConfig['outputNetworkFile']}")
        self.map_schedule(mapper_config_path)


class MATSimNetworkCreator:
    def __init__(self, config, engine: NetworkCreatorEngine, engineArgs={},log_level=Logger.Level.SILENT):
        self.config = config
        self.logger = Logger(log_level)
        self._run = lambda cmd: __my_run__(self,cmd)
        self.clean_tmp = config.get("clean_tmp", True)
        Path(".tmp").mkdir(parents=True, exist_ok=True)
        self.check_requirements()
        self.engine = engine(logger=self.logger, **engineArgs)

    def check_requirements(self):
        if shutil.which("osmium") is None:
            if self.config.get("auto_install_requirements", False):
                self.logger.info("Missing osmium, will install it")
                self._run(["sudo", "apt", "install", "osmium-tool"])
            else:
                raise RequirementError("Missing osmium tool")

    def __cleanup_tmp(self):
        tmp = Path(".tmp")
        if tmp.exists():
            self.logger.info(f"Removing .tmp folder")
            shutil.rmtree(tmp)

    def download_file(self, url, output_path, skip_if_exists=True):
        if skip_if_exists and Path(output_path).exists():
            self.logger.info(f"Skipping download {output_path}")
            return
        self.logger.info(f"Downloading {output_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

    def pbf_to_osm(self, input_path, output_path, skip_if_exists=True):
        if skip_if_exists and Path(output_path).exists():
            self.logger.info(f"Skipping pbf to osm {input_path}")
            return
        self.logger.info(f"Converting pbf to osm: {input_path} -> {output_path}")
        self._run(["osmium", "cat", input_path, "-o", output_path])

    def crop_osm(self, input_path, output_path, bbox, skip_if_exists=True):
        if skip_if_exists and Path(output_path).exists():
            self.logger.info(f"Skipping cropping to {output_path}")
            return
        self.logger.info(f"Cropping: {input_path} -> {output_path}")
        self._run(["osmium", "extract", "-b", ",".join(map(str, bbox)), input_path, "-o", output_path])

    def crop_osm_relation(self, input_path, output_path, relation_id, skip_if_exists=True):
        if skip_if_exists and Path(output_path).exists():
            self.logger.info(f"Skipping cropping to {output_path}")
            return
        self.logger.info(f"Cropping: {input_path} -> {output_path}")
        self._run(["osmium", "getid", "-r", "-t", input_path, f"r{relation_id}", "-o", ".tmp/boundary.osm"])
        self._run(["osmium", "extract", "--polygon", ".tmp/boundary.osm", input_path, "-o", output_path])

    def create_network(self, engineArgs):
        osm_path = self.config.get("osm_download_path",".tmp/map_full.osm.pbf")
        self.download_file(self.config["osm_url"], osm_path, self.config.get("skip_downloads", False))
        
        if "osm_crop_bbox" in self.config:
            crop_path = self.config.get("osm_crop_path",".tmp/map_cropped.osm")
            self.crop_osm(osm_path, crop_path, self.config["osm_crop_bbox"], self.config.get("skip_cropping", False))
            osm_path = crop_path
        
        if "osm_crop_relation" in self.config:
            crop_path = self.config.get("osm_crop_path",".tmp/map_cropped.osm")
            self.crop_osm_relation(osm_path, crop_path, self.config["osm_crop_relation"], self.config.get("skip_cropping", False))
            osm_path = crop_path

        if osm_path[-4:] == ".pbf":
            self.pbf_to_osm(osm_path, osm_path[:-4], self.config.get("skip_downloads", False))
            osm_path = osm_path[:-4]

        if "gtfs_url" in self.config:
            gtfs_path = self.config.get("gtfs_download_path",".tmp/gtfs.zip")
            self.download_file(self.config["gtfs_url"], gtfs_path, self.config.get("skip_downloads", False))
            self.config["gtfs_path"] = self.config.get("gtfs_path", gtfs_path)

        self.engine.createNetwork(osm_path, self.config.get("output_network_path","network.xml"), *engineArgs)
        if self.clean_tmp:
            self.__cleanup_tmp()