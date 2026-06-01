import csv
import yaml
import matsim
import random
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from time import time as now
import matplotlib.pyplot as plt
from collections import defaultdict
from shapely.geometry import Polygon
from bisect import bisect_left as lower_bound

def parse_vehicle_type(vt):
    if type(vt) == str:
        return vt
    if type(vt) == dict and "chance" in vt:
        options = list(vt["chance"].items())
        return random.choices([x[0] for x in options], [x[1] for x in options], k=1)[0]
    raise ValueError(f"Invalid vehicle type: {vt}")

def _classify_vehicle(vehicle_id, config):
    vehicle_id = str(vehicle_id).lower()

    if vehicle_id in config["vehicle id to type"]:
        return config["vehicle id to type"][vehicle_id]

    includes = [k for k in config["vehicle id to type"].keys() if k.startswith("includes ")]

    for include in includes:
        if include[len("includes "):] in vehicle_id:
            return config["vehicle id to type"][include]

    return config["vehicle id to type"]["default"]

def classify_vehicle(vehicle_id, config):
    return parse_vehicle_type(_classify_vehicle(vehicle_id, config))

def get_emission_factors(speed, vehicle_type, config):

    bins = config["vehicle_speed_bins"][vehicle_type]

    idx = lower_bound(bins[0], speed)

    return {pollutant: (value,unit) for pollutant, value, unit in bins[2][idx] }

def get_ef_array(speed, vehicle_type, ef_lookup):
    speed_bins, arrays = ef_lookup[vehicle_type]
    idx = lower_bound(speed_bins, speed)
    return arrays[idx] 

def get_grid_cell(x, y, grid_size):
    
    gx = int(x // grid_size) * grid_size
    gy = int(y // grid_size) * grid_size

    return (gx, gy)

def load_config(path):
    with open(path) as f:
        config_str = f.read()
    return parse_config(config_str)

def parse_config(config_str):
    config = yaml.safe_load(config_str)
    
    if "seed" in config:
        random.seed(config["seed"])
        np.random.seed(config["seed"])

    time_bin = 0
    time_str_split = config["time_bin_size"] = config["time_bin_size"].lower().strip().split()
    for i in range(0, len(time_str_split), 2):
        value = int(time_str_split[i])
        unit = time_str_split[i+1]
        if unit.startswith("h"):
            time_bin += value * 3600
        elif unit.startswith("m"):
            time_bin += value * 60
        elif unit.startswith("s"):
            time_bin += value
        else:
            raise ValueError(f"Unknown time unit: {unit}")
    config["time_bin_size"] = time_bin

    pollutants = set()
    speed_bins = defaultdict(list)
    emission_yaml_key = "emission by vehicle"
    for vehicle_type, data in config[emission_yaml_key].items():
        speed_bin = []
        polutant_bin = []
        unit_bin = []
        for key, value in data.items():
            if key.startswith("up to"):
                max_speed = float(key.split()[2])
                speed_unit = key.split()[3]
            if key.startswith("above"):
                max_speed = float("inf")
                speed_unit = key.split()[2]
            vehicle_pollutants = []
            for pollutant in value.split(","):
                value, unit, name = pollutant.strip().split()
                vehicle_pollutants.append((name, float(value), unit))
                pollutants.add(name)
            speed_bin.append(max_speed)
            polutant_bin.append(vehicle_pollutants)
            unit_bin.append(speed_unit)
        
        if not 0 in speed_bin:
            speed_bin.append(0)
            unit_bin.append(speed_unit)
            polutant_bin.append([(name, 0, unit) for name in pollutants])

        combined = sorted(zip(speed_bin, unit_bin, polutant_bin), key=lambda x: x[0])
        speed_bin, unit_bin, polutant_bin = zip(*combined)
        speed_bins[vehicle_type] = (speed_bin, unit_bin, polutant_bin)
    
    config["vehicles"] = list(config[emission_yaml_key].keys())

    del config[emission_yaml_key]
    config["pollutants"] = list(pollutants)
    config["vehicle_speed_bins"] = speed_bins
    return config

def main(config_path="emission_config.yaml"):
    print("Loading configuration...")
    config = load_config(config_path)

    print("Loading network...")
    network = matsim.read_network(config["network_file"])

    print("Setting up data structures...")
    network.links["idx"] = network.links.index
    network.links["x"] = (network.nodes.set_index("node_id").loc[network.links["from_node"]]["x"].values + network.nodes.set_index("node_id").loc[network.links["to_node"]]["x"].values) / 2
    network.links["y"] = (network.nodes.set_index("node_id").loc[network.links["from_node"]]["y"].values + network.nodes.set_index("node_id").loc[network.links["to_node"]]["y"].values) / 2
    links= network.links.set_index("link_id")

    link_ids_to_int = links["idx"].to_dict()
    links_int_to_id = {v:k for k,v in link_ids_to_int.items()}
    link_centers = {k: (v["x"],v["y"]) for k, v in links[["x","y"]].to_dict(orient="index").items()}
    link_lengths = links["length"].to_dict()
    grid_to_index = {k:v for v,k in enumerate(list({get_grid_cell(x,y, config["grid_size_m"]) for x,y in link_centers.values()}))}
    pollutant_index = {p: i for i, p in enumerate(config["pollutants"])}
    vehicle_type_to_idx = {v: i for i, v in enumerate(config["vehicles"])}

    time_bin_size = config["time_bin_size"]
    polutants_size = len(config["pollutants"])
    links_size = len(link_ids_to_int)
    time_bins_amount = 24*3600//time_bin_size
    temporal_emissions = np.zeros((time_bins_amount, polutants_size+2))
    link_emissions = np.zeros((links_size, polutants_size+3))
    grid_emissions = np.zeros((len(grid_to_index)*time_bins_amount, polutants_size+4))
    vehicle_emissions = np.zeros((len(config["vehicles"]), polutants_size+3))

    ef_lookup = {}
    for vehicle_type, (speed_bin, unit_bin, pollutant_bin) in config["vehicle_speed_bins"].items():
        arrays = []
        for bin_pollutants in pollutant_bin:
            arr = np.zeros(len(config["pollutants"]))
            for name, value, unit in bin_pollutants:
                arr[pollutant_index[name]] = value
            arrays.append(arr)
        ef_lookup[vehicle_type] = (speed_bin, arrays)

    events = matsim.event_reader(
        config["events_file"],
        types=["entered link", "left link"]
    )

    active_links = {}

    vehicle_type_cache = {}

    counter = 0
    counter_threshold = 1000
    counter_threshold_growth = 1.2
    noks = 0
    total = 0

    start = now()

    print("Processing events...")
    for event in events:

        counter += 1

        if counter >= counter_threshold:
            perc_hour = event['time']/3600
            remaining_seconds =  (now() - start)/(perc_hour/24)
            remaining_time = datetime.timedelta(seconds=int(remaining_seconds))
            print(f"\tProcessed {counter} events, currently at: {perc_hour:.2f} hours into simulation, progress: {perc_hour/24*100:.2f}%, estimated remaining time: {remaining_time}")
            counter_threshold *= 1 + counter_threshold_growth
            

        event_type = event["type"]

        vehicle_id = event["vehicle"]
        link_id = event["link"]
        time = float(event["time"])

        if event_type == "entered link":
            active_links[(vehicle_id, link_id)] = time

        elif event_type == "left link":
            
            total += 1
            if (vehicle_id, link_id) not in active_links:
                #print(f"Vehicle {vehicle_id} left link {link_id} without entering it!?")
                noks += 1
                continue
                
            enter_time = active_links[(vehicle_id, link_id)]

            travel_time = time - enter_time

            if travel_time <= 0:
                raise ValueError(f"Vehicle {vehicle_id} has non-positive travel time on link {link_id} (enter time: {enter_time}, leave time: {time})")

            link_length = link_lengths[link_id]

            distance_km = link_length / 1000.0

            speed_kmh = (distance_km / (travel_time / 3600.0))

            vehicle_type = vehicle_type_cache.get(vehicle_id, classify_vehicle(vehicle_id, config))

            ef_array = get_ef_array(speed_kmh, vehicle_type, ef_lookup)

            time_bin = int(time // time_bin_size)
        
            x,y = link_centers[link_id]
            cell = get_grid_cell(x,y,config["grid_size_m"])
            cell_idx = grid_to_index[cell] + time_bin * len(grid_to_index)

            link_emissions_idx = link_ids_to_int[link_id]
            
            emission_col_slice = slice(1, polutants_size + 1)
            
            delta = ef_array * distance_km

            vehicle_type_idx = vehicle_type_to_idx[vehicle_type]
            vehicle_emissions[vehicle_type_idx, emission_col_slice] += delta
            vehicle_emissions[vehicle_type_idx, -1] += 1
            vehicle_emissions[vehicle_type_idx, -2] += link_length
            vehicle_emissions[vehicle_type_idx, 0] = vehicle_type_idx

            link_emissions[link_emissions_idx, emission_col_slice] += delta
            link_emissions[link_emissions_idx, -1] += 1
            link_emissions[link_emissions_idx, 0] = link_emissions_idx
            link_emissions[link_emissions_idx, -2] += link_length

            temporal_emissions[time_bin, emission_col_slice] += delta
            temporal_emissions[time_bin, -1] += 1
            temporal_emissions[time_bin, 0] = time_bin

            emission_col_slice = slice(3, polutants_size + 3)
            grid_emissions[cell_idx, emission_col_slice] += delta
            grid_emissions[cell_idx, -1] += 1
            grid_emissions[cell_idx, 0] = cell[0]
            grid_emissions[cell_idx, 1] = cell[1]
            grid_emissions[cell_idx, 2] = time_bin

            del active_links[(vehicle_id, link_id)]
    
    Path(config["output_folder"]).mkdir(exist_ok=True)

    vehicle_df = pd.DataFrame(vehicle_emissions, columns=["vehicle_type"] + config["pollutants"] + ["distance_m", "vehicles"])
    vehicle_df = vehicle_df[~(vehicle_df == 0).all(axis=1)]
    vehicle_df["vehicle_type"] = vehicle_df["vehicle_type"].apply(lambda x: config["vehicles"][int(x)])
    vehicle_df.to_csv(Path(config["output_folder"]) / "vehicle_emissions.csv", index=False)

    link_df = pd.DataFrame(link_emissions, columns=["link_id"] + config["pollutants"] + ["distance_m", "vehicles"])
    link_df = link_df[~(link_df == 0).all(axis=1)]
    link_df["link_id"] = link_df["link_id"].apply(lambda x: links_int_to_id[x])
    link_df.to_csv(Path(config["output_folder"]) / "link_emissions.csv", index=False)

    temporal_df = pd.DataFrame(temporal_emissions, columns=["time_bin"] + config["pollutants"] + ["vehicles"])
    temporal_df = temporal_df[~(temporal_df == 0).all(axis=1)]
    temporal_df["seconds"] = temporal_df["time_bin"].apply(lambda x: (x+1) * time_bin_size)
    temporal_df = temporal_df[["time_bin", "seconds"]+config["pollutants"] + ["vehicles"]]
    temporal_df.to_csv(Path(config["output_folder"]) / "temporal_emissions.csv", index=False)

    grid_df = pd.DataFrame(grid_emissions, columns=["x", "y", "time_bin"] + config["pollutants"] + ["vehicles"])
    grid_df = grid_df[~(grid_df[config["pollutants"]] == 0).all(axis=1)]
    grid_df["geometry"] = grid_df.apply(lambda row: Polygon([(row["x"], row["y"]), (row["x"]+config["grid_size_m"], row["y"]), (row["x"]+config["grid_size_m"], row["y"]+config["grid_size_m"]), (row["x"], row["y"]+config["grid_size_m"])]), axis=1)
    grid_df["i"] = grid_df["x"].apply(lambda x: sorted(grid_df["x"].unique()).index(x))
    grid_df["j"] = grid_df["y"].apply(lambda y: sorted(grid_df["y"].unique()).index(y))
    grid_df = grid_df[["i","x", "j","y", "time_bin"] + config["pollutants"] + ["vehicles", "geometry"]]
    grid_df.sort_values(["time_bin", "i", "j"], inplace=True)
    gdf = gpd.GeoDataFrame(grid_df)
    if "crs" in config: gdf.crs = config["crs"]
    gdf.to_file(
        Path(config["output_folder"]) / "spatial_emissions.geojson",
        driver="GeoJSON"
    )
    gdf.to_csv(Path(config["output_folder"]) / "spatial_emissions.csv", index=False)

if __name__ == "__main__":
    main()