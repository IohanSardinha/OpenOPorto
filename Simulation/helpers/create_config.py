from extract_config_info import extract_plan_info,extract_vehicle_info
from oporto_default_config import DEFAULT_CONFIG
from pathlib import Path
import importlib.util
import argparse

def load_config(path):
    path = Path(path).resolve()

    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.config

def main():
    parser = argparse.ArgumentParser(description="Creates a default MATSim config file with activity parameters and transit modes extracted from the population and vehicles files.")
    parser.add_argument("output", help="Output config file path", nargs="?", default="input/config.xml")
    parser.add_argument("config", help="Path to the config.py file", nargs="?", default="config.py")
    args = parser.parse_args()

    config = load_config(args.config)

    config["transitModes"] = extract_vehicle_info(config["inputFolder"] + config["vehiclesFile"])
    _,_, plans_info, modes = extract_plan_info(config["inputFolder"] + config["inputPlansFile"])

    config["activityParams"] = [{"type": k,
                                "typicalDuration": v["avg"]
                                } for k,v in plans_info.items()]
    config["activityModes"] = modes

    with open(args.output, "w") as f:
        f.write(DEFAULT_CONFIG(config))

if __name__ == "__main__":
    main()