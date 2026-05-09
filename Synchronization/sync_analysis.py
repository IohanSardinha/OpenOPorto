"""
Transit Line Synchronization Analysis Module

This module analyzes synchronization between transit lines in MATSim simulations.
It processes transit schedules, events, and passenger legs to compute synchronization
metrics and generate CSV reports.
"""

import matsim
import xml.etree.ElementTree as ET
import gzip
import pandas as pd
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
import warnings


class SyncAnalyzer:
    """Analyzes synchronization between transit lines."""
    
    def __init__(self, simulation_output_folder: str, sync_lines: List[str], 
                 sync_nodes: List[List[Tuple[str, str]]], output_folder: str = "results/"):
        """
        Initialize the synchronization analyzer.
        
        Args:
            simulation_output_folder: Path to MATSim simulation output folder
            sync_lines: List of transit line IDs to analyze (e.g., ["A", "D"])
            sync_nodes: List of synchronization node pairs
                       Format: [[("A","stop1"), ("D","stop2")], ...]
            output_folder: Path where CSV results will be saved
        """
        self.sim_folder = Path(simulation_output_folder)
        self.sync_lines = sync_lines
        self.sync_nodes = sync_nodes
        self.output_folder = Path(output_folder)
        
        # Validate inputs
        self._validate_inputs()
        
        # Create output directories
        self._create_output_dirs()
        
        # Data structures that will be populated
        self.sync_stops = {}
        self.route_to_syncline = {}
        self.veh_to_syncline = {}
        self.syncline_stops = {}
        self.syncline_to_line = {}
        self.nsync_lines = []
        self.stop_names = {}
        self.syncline_departures = {}
        
        # Metrics
        self.H = {}  # Headways: H[(i, p)]
        self.T = {}  # Travel times: T[(i, s, p)]
        self.d = {}  # Delays: d[(i, s, p)]
        self.N = {}  # Transfers: N[(i, j, s)]
    
    def _validate_inputs(self):
        """Validate input parameters."""
        # Check simulation folder exists
        if not self.sim_folder.exists():
            raise FileNotFoundError(f"Simulation folder not found: {self.sim_folder}")
        
        # Check required files exist
        required_files = [
            "output_transitSchedule.xml.gz",
            "output_events.xml.gz",
            "output_legs.csv"
        ]
        for file in required_files:
            file_path = self.sim_folder / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Validate sync_nodes format
        if not self.sync_nodes:
            raise ValueError("sync_nodes cannot be empty")
        
        for i, cluster in enumerate(self.sync_nodes):
            if len(cluster) != 2:
                raise ValueError(f"sync_nodes[{i}] must contain exactly 2 tuples, got {len(cluster)}")
            for j, node in enumerate(cluster):
                if len(node) != 2:
                    raise ValueError(f"sync_nodes[{i}][{j}] must be a 2-tuple (line_id, stop_id)")
    
    def _create_output_dirs(self):
        """Create output directories if they don't exist."""
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder / "complete").mkdir(exist_ok=True)
    
    def _build_sync_stops(self):
        """Build syncStops dictionary from syncNodes."""
        for cluster in self.sync_nodes:
            for line, stop in cluster:
                self.sync_stops[line] = self.sync_stops.get(line, []) + [stop]
    
    def _parse_schedule(self):
        """Parse transit schedule and build route mappings."""
        print("Parsing transit schedule...")
        
        with gzip.open(self.sim_folder / "output_transitSchedule.xml.gz", "r") as f:
            tree = ET.parse(f)
            root = tree.getroot()
        
        # Extract stop names
        self.stop_names = {
            stop.get("id"): stop.get("name") 
            for stop in root.findall(".//stopFacility")
        }
        
        # Process transit lines
        counts = {}
        for transit_line in root.findall("transitLine"):
            line_id = transit_line.get("id")
            
            if line_id not in self.sync_lines:
                continue
            
            for transit_route in transit_line.findall("transitRoute"):
                # Build route signature from stops
                route_profile = transit_route.find("routeProfile")
                if route_profile is None:
                    warnings.warn(f"No routeProfile found for line {line_id}")
                    continue
                
                route_stops = [stop.get("refId") for stop in route_profile.findall("stop")]
                route_signature = "|".join(route_stops)
                
                # Create unique syncline identifier if route is new
                if route_signature not in self.route_to_syncline:
                    counts[line_id] = counts.get(line_id, 0) + 1
                    syncline_id = f"{line_id}_{counts[line_id]}"
                    self.route_to_syncline[route_signature] = syncline_id
                    self.syncline_stops[syncline_id] = route_stops
                    self.syncline_to_line[syncline_id] = line_id
                
                syncline_id = self.route_to_syncline[route_signature]
                
                # Map vehicles to synclines
                for departure in transit_route.findall(".//departure"):
                    veh_id = departure.get("vehicleRefId")
                    self.veh_to_syncline[veh_id] = syncline_id
        
        self.nsync_lines = list(self.route_to_syncline.values())
        
        if not self.nsync_lines:
            raise ValueError(f"No routes found for sync lines: {self.sync_lines}")
        
        print(f"Found {len(self.nsync_lines)} unique routes across {len(self.sync_lines)} lines")
    
    def _process_events(self):
        """Process events to compute headways, travel times, and delays."""
        print("Processing events...")
        
        events = matsim.event_reader(
            str(self.sim_folder / "output_events.xml.gz"),
            types=["TransitDriverStarts", "VehicleArrivesAtFacility", "VehicleDepartsAtFacility"]
        )
        
        start_times = {}
        line_counter = {}
        veh_to_trip_number = {}
        last_departure = {}
        
        process_count = 0
        process_threshold = 10000
        process_threshold_growth = 10000
        for event in events:
            process_count += 1
            if process_count>= process_threshold:
                print(f"\tProcessed {process_count} events, currently at: {event['time']/3600:.2f} hours into simulation")
                process_threshold_growth *= 1.2
                process_threshold += process_threshold_growth

            veh_id = event.get("vehicleId") or event.get("vehicle")
            
            if event["type"] == "TransitDriverStarts":
                syncline = self.veh_to_syncline.get(veh_id)
                if syncline not in self.nsync_lines:
                    continue
                
                start_times[veh_id] = event["time"]
                line_counter[syncline] = line_counter.get(syncline, 0) + 1
                veh_to_trip_number[veh_id] = line_counter[syncline]
                
                i = self.nsync_lines.index(syncline)
                p = veh_to_trip_number[veh_id]
                
                # Calculate headway
                if i in last_departure:
                    self.H[(i, p)] = event["time"] - last_departure[i]
                else:
                    self.H[(i, p)] = 0
                
                last_departure[i] = event["time"]
            
            elif event["type"] == "VehicleArrivesAtFacility":
                syncline = self.veh_to_syncline.get(veh_id)
                if syncline not in self.nsync_lines:
                    continue
                
                i = self.nsync_lines.index(syncline)
                facility = event["facility"]
                
                # Find stop index in route
                if facility not in self.syncline_stops[syncline]:
                    continue
                
                s = self.syncline_stops[syncline].index(facility)
                p = veh_to_trip_number[veh_id]
                
                # Calculate travel time and delay
                self.T[(i, s, p)] = event["time"] - start_times[veh_id]
                self.d[(i, s, p)] = float(event["delay"])
        
        # Count departures per syncline
        self.syncline_departures = {
            syncline: len([v for v, l in self.veh_to_syncline.items() if l == syncline])
            for syncline in self.nsync_lines
        }
        
        print(f"Processed {sum(self.syncline_departures.values())} vehicle departures in {process_count} events")
    
    def _process_transfers(self):
        """Process passenger legs to count transfers."""
        print("Processing transfers...")
        
        legs = pd.read_csv(self.sim_folder / "output_legs.csv", sep=";")
        
        transfers = {}
        
        for (person, trip_id), trip_legs in legs.groupby(["person", "trip_id"], sort=False):
            trip_legs = trip_legs.reset_index(drop=True)
            pt_positions = trip_legs.index[trip_legs["mode"] == "pt"].tolist()
            
            if len(pt_positions) < 2:
                continue
            
            for prev_i, next_i in zip(pt_positions[:-1], pt_positions[1:]):
                # Only consider direct transfers (adjacent PT legs)
                if next_i - prev_i != 2:
                    continue
                
                prev_pt = trip_legs.loc[prev_i]
                next_pt = trip_legs.loc[next_i]
                
                # Check if both legs are on sync lines
                if (prev_pt["transit_line"] not in self.sync_lines or 
                    next_pt["transit_line"] not in self.sync_lines):
                    continue
                
                prev_veh = prev_pt["vehicle_id"]
                prev_stop = prev_pt["egress_stop_id"]
                prev_syncline = self.veh_to_syncline.get(prev_veh)
                
                next_veh = next_pt["vehicle_id"]
                next_stop = next_pt["access_stop_id"]
                next_syncline = self.veh_to_syncline.get(next_veh)
                
                if not prev_syncline or not next_syncline:
                    continue
                
                key = (prev_syncline, prev_stop, next_syncline, next_stop)
                if key not in transfers:
                    transfers[key] = set()
                transfers[key].add((person, next_pt["dep_time"]))
        
        # Convert to N dictionary
        for (prev_syncline, prev_stop, next_syncline, next_stop), transfer_set in transfers.items():
            try:
                i = self.nsync_lines.index(prev_syncline)
                j = self.nsync_lines.index(next_syncline)
                
                # Find the sync node index
                s = None
                for idx, cluster in enumerate(self.sync_nodes):
                    stops = [stop for _, stop in cluster]
                    if prev_stop in stops and next_stop in stops:
                        s = idx
                        break
                
                if s is not None:
                    self.N[(i, j, s)] = len(transfer_set)
            except (ValueError, IndexError) as e:
                warnings.warn(f"Could not process transfer: {e}")
        
        total_transfers = sum(self.N.values())
        print(f"Found {total_transfers} transfers across {len(self.N)} OD pairs")
    
    def run_analysis(self):
        """Run the complete synchronization analysis."""
        print("=" * 60)
        print("Starting Synchronization Analysis")
        print("=" * 60)
        
        # Build sync stops mapping
        self._build_sync_stops()
        
        # Parse schedule
        self._parse_schedule()
        
        # Process events
        self._process_events()
        
        # Process transfers
        self._process_transfers()
        
        # Generate reports
        self._generate_reports()
        
        print("=" * 60)
        print("Analysis Complete!")
        print(f"Results saved to: {self.output_folder.absolute()}")
        print("=" * 60)
    
    def _generate_reports(self):
        """Generate all CSV reports."""
        print("\nGenerating CSV reports...")
        
        self._generate_lines_csv()
        self._generate_sync_nodes_csv()
        self._generate_sync_nodes_matrix_csv()
        self._generate_sync_nodes_summary_csv()
        self._generate_transfers_csv()
        self._generate_schedule_deviations_csv()
        self._generate_travel_times_csv()
        self._generate_headways_csv()
        
        print("All reports generated successfully")
    
    def _generate_lines_csv(self):
        """Generate lines.csv"""
        lines = list(range(1, len(self.nsync_lines) + 1))
        ids = self.nsync_lines
        names = [self.syncline_to_line[line] for line in self.nsync_lines]
        origins = [self.syncline_stops[line][0] for line in self.nsync_lines]
        origin_names = [self.stop_names.get(o, o) for o in origins]
        destinations = [self.syncline_stops[line][-1] for line in self.nsync_lines]
        dest_names = [self.stop_names.get(d, d) for d in destinations]
        
        # Calculate min/max headways
        hi_min = []
        hi_max = []
        for i in range(len(self.nsync_lines)):
            headways = [h for (idx, p), h in self.H.items() if idx == i and h != 0]
            hi_min.append(min(headways) if headways else 0)
            hi_max.append(max(headways) if headways else 0)
        
        # Complete version
        complete_headers = ["Line", "ID", "Name", "Origin_id", "Origin_Name", 
                           "Destination_id", "Destination_Name", "h(i)Min", "h(i)Max"]
        complete_df = pd.DataFrame(list(zip(
            lines, ids, names, origins, origin_names, 
            destinations, dest_names, hi_min, hi_max
        )), columns=complete_headers)
        complete_df.to_csv(self.output_folder / "complete" / "lines.csv", index=False)
        
        # Short version
        short_headers = ["Line", "Origin", "Destination", "h(i)Min", "h(i)Max"]
        short_df = pd.DataFrame(list(zip(
            lines, origin_names, dest_names, hi_min, hi_max
        )), columns=short_headers)
        short_df.to_csv(self.output_folder / "lines.csv", index=False)
    
    def _generate_sync_nodes_csv(self):
        """Generate sync_nodes.csv"""
        node_ids = [f"Node {i}" for i in range(1, len(self.sync_nodes) + 1)]
        line_i = [n[0] for n, _ in self.sync_nodes]
        id_line_i = [n[1] for n, _ in self.sync_nodes]
        name_line_i = [self.stop_names.get(n[1], n[1]) for n, _ in self.sync_nodes]
        line_j = [n[0] for _, n in self.sync_nodes]
        id_line_j = [n[1] for _, n in self.sync_nodes]
        name_line_j = [self.stop_names.get(n[1], n[1]) for _, n in self.sync_nodes]
        
        # Complete version
        complete_headers = ["Nodeid", "Line_i", "ID_Line_i", "Name_Line_i", 
                           "Line_j", "ID_Line_j", "Name_Line_j"]
        complete_df = pd.DataFrame(list(zip(
            node_ids, line_i, id_line_i, name_line_i, 
            line_j, id_line_j, name_line_j
        )), columns=complete_headers)
        complete_df.to_csv(self.output_folder / "complete" / "sync_nodes.csv", index=False)
        
        # Short version
        short_df = pd.DataFrame(list(zip(node_ids, name_line_i)))
        short_df.to_csv(self.output_folder / "sync_nodes.csv", index=False, header=False)
    
    def _generate_sync_nodes_matrix_csv(self):
        """Generate sync_nodes_matrix.csv"""
        lines = list(range(1, len(self.nsync_lines) + 1))
        
        # Build intersections dictionary (simplified - takes first match)
        intersections = {cluster[0][0]: cluster[1] for cluster in self.sync_nodes}
        
        # Build matrix
        matrix = []
        for i in lines:
            row = [i]
            for j in lines:
                syncline_i = self.nsync_lines[i - 1]
                line_i = self.syncline_to_line[syncline_i]
                
                intersects = intersections.get(line_i, None)
                
                syncline_j = self.nsync_lines[j - 1]
                line_j = self.syncline_to_line[syncline_j]
                
                if intersects and line_j == intersects[0]:
                    stop_name = self.stop_names.get(intersects[1], intersects[1])
                    intersect_val = (stop_name, intersects[1])
                else:
                    intersect_val = ""
                
                row.append(intersect_val)
            matrix.append(row)
        
        # Complete version with full details
        complete_matrix = deepcopy(matrix)
        complete_headers = ["Lines"] + [f"{self.nsync_lines[i-1]}({i})" for i in lines]
        for line in complete_matrix:
            line[0] = f"{self.nsync_lines[line[0]-1]} ({line[0]})"
            for i in range(1, len(line)):
                if line[i] != "":
                    line[i] = f"{line[i][0]} ({line[i][1]})"
        
        complete_df = pd.DataFrame(complete_matrix, columns=complete_headers)
        complete_df.to_csv(self.output_folder / "complete" / "sync_nodes_matrix.csv", index=False)
        
        # Short version with just stop names
        short_headers = ["Lines"] + lines
        for line in matrix:
            for i in range(1, len(line)):
                if line[i] != "":
                    line[i] = line[i][0]
        
        short_df = pd.DataFrame(matrix, columns=short_headers)
        self.snm = short_df 
        short_df.to_csv(self.output_folder / "sync_nodes_matrix.csv", index=False)
    
    def _generate_sync_nodes_summary_csv(self):
        """Generate sync_nodes_summary.csv"""
        # Read the matrix we just created
        lines = list(range(1, len(self.nsync_lines) + 1))
        
        # Convert to line numbers
        ll = self.snm.apply(
            lambda row: [lines[i-1] if isinstance(x, str) and x != "" else x 
                        for i, x in enumerate(row)], 
            axis=1, 
            result_type="expand"
        )
        
        # Keep only columns with data
        ll = ll.loc[:, (ll != "").any(axis=0)]
        
        # Calculate I_i (number of intersecting lines)
        I_i = ll.apply(lambda row: sum(1 for x in row[1:] if x != ""), axis=1)
        
        # Calculate F_i (max departures among intersecting lines)
        F_i = ll.apply(
            lambda row: max(
                [self.syncline_departures[self.nsync_lines[x-1]] 
                 for x in row if isinstance(x, int)], 
                default=0
            ), 
            axis=1
        )
        
        # Rename columns
        ll.rename(columns={i: "" for i in ll.columns if i != 0}, inplace=True)
        ll.rename(columns={0: "Lines"}, inplace=True)
        ll["I_i"] = I_i
        ll["F_i"] = F_i
        
        ll.to_csv(self.output_folder / "sync_nodes_summary.csv", index=False)
    
    def _generate_transfers_csv(self):
        """Generate number_of_transfers.csv"""
        headers = ["i", "j", "s", "N(i,s,j)"]
        Is, Js, Ss, Ns = [], [], [], []
        
        for i in range(len(self.nsync_lines)):
            for j in range(len(self.nsync_lines)):
                for s in range(len(self.sync_nodes)):
                    Is.append(f"Line {i+1}" if j == 0 else "")
                    Js.append(j + 1)
                    Ss.append(s + 1)
                    Ns.append(self.N.get((i, j, s), ""))
        
        df = pd.DataFrame(list(zip(Is, Js, Ss, Ns)), columns=headers)
        df.to_csv(self.output_folder / "number_of_transfers.csv", index=False)
    
    def _generate_schedule_deviations_csv(self):
        """Generate schedule_deviations.csv"""
        headers = ["i", "s", "p", "d(i,s,p)"]
        Is, Ss, Ps, Ds = [], [], [], []
        
        p_max = max(self.syncline_departures.values())
        
        for i, line_i in enumerate(self.nsync_lines):
            for s in range(len(self.sync_nodes)):
                # Get the actual stop in this line's route
                line_id = self.syncline_to_line[line_i]
                sync_stop = self.sync_stops[line_id][s]
                
                if sync_stop not in self.syncline_stops[line_i]:
                    continue
                
                s_lookup = self.syncline_stops[line_i].index(sync_stop)
                
                for p in range(1, p_max + 1):
                    Is.append(f"Line {i+1}" if p == 1 else "")
                    Ss.append(s + 1)
                    Ps.append(p)
                    
                    delay = self.d.get((i, s_lookup, p), "")
                    Ds.append(round(delay / 60) if isinstance(delay, float) else "")
        
        df = pd.DataFrame(list(zip(Is, Ss, Ps, Ds)), columns=headers)
        df.to_csv(self.output_folder / "schedule_deviations.csv", index=False)
    
    def _generate_travel_times_csv(self):
        """Generate travel_times.csv"""
        headers = ["i", "s", "p", "T(i,s,p)"]
        Is, Ss, Ps, Ts = [], [], [], []
        
        p_max = max(self.syncline_departures.values())
        
        for i, line_i in enumerate(self.nsync_lines):
            for s in range(len(self.sync_nodes)):
                # Get the actual stop in this line's route
                line_id = self.syncline_to_line[line_i]
                sync_stop = self.sync_stops[line_id][s]
                
                if sync_stop not in self.syncline_stops[line_i]:
                    continue
                
                s_lookup = self.syncline_stops[line_i].index(sync_stop)
                
                for p in range(1, p_max + 1):
                    Is.append(f"Line {i+1}" if p == 1 else "")
                    Ss.append(s + 1)
                    Ps.append(p)
                    
                    travel_time = self.T.get((i, s_lookup, p), "")
                    Ts.append(round(travel_time / 60) if isinstance(travel_time, float) else "")
        
        df = pd.DataFrame(list(zip(Is, Ss, Ps, Ts)), columns=headers)
        df.to_csv(self.output_folder / "travel_times.csv", index=False)
    
    def _generate_headways_csv(self):
        """Generate headways.csv"""
        headers = ["i", "p", "h(i,p)"]
        Is, Ps, Hs = [], [], []
        
        p_max = max(self.syncline_departures.values())
        
        for i in range(len(self.nsync_lines)):
            for p in range(1, p_max + 1):
                Is.append(f"Line {i+1}" if p == 1 else "")
                Ps.append(p)
                
                headway = self.H.get((i, p), "") if p > 1 else 0.0
                Hs.append(round(headway / 60) if isinstance(headway, float) else "")
        
        df = pd.DataFrame(list(zip(Is, Ps, Hs)), columns=headers)
        df.to_csv(self.output_folder / "headways.csv", index=False)


def run_sync_analysis(simulation_folder: str, sync_lines: List[str], 
                      sync_nodes: List[List[Tuple[str, str]]], 
                      output_folder: str = "results/") -> SyncAnalyzer:
    """
    Convenience function to run synchronization analysis.
    
    Args:
        simulation_folder: Path to MATSim simulation output
        sync_lines: List of line IDs to synchronize
        sync_nodes: List of synchronization node pairs
        output_folder: Where to save results
    
    Returns:
        SyncAnalyzer instance with results
    
    Example:
        >>> analyzer = run_sync_analysis(
        ...     "../Simulation/output/",
        ...     ["A", "D"],
        ...     [[("A", "5726.link:pt_5726"), ("D", "5726.link:pt_5726")]]
        ... )
    """
    analyzer = SyncAnalyzer(simulation_folder, sync_lines, sync_nodes, output_folder)
    analyzer.run_analysis()
    return analyzer


if __name__ == "__main__":

    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run synchronization analysis on MATSim output"
    )

    # Positional config argument
    parser.add_argument(
        "config",
        nargs="?",
        help="Path to JSON config file with parameters"
    )

    # Optional flag
    parser.add_argument(
        "--generate-example-config",
        action="store_true",
        help="Generate an example config file"
    )

    args = parser.parse_args()

    # Generate example config
    if args.generate_example_config:
        example_config = {
            "simulation_folder": "../Simulation/output/",
            "sync_lines": ["A", "D"],
            "sync_nodes": [
                [("A", "5726.link:pt_5726"),
                ("D", "5726.link:pt_5726")]
            ],
            "output_folder": "results/"
        }

        with open("example_config.json", "w") as f:
            json.dump(example_config, f, indent=4)

        print("Example config file generated: example_config.json")
        exit(0)

    # No config provided
    if not args.config:
        parser.error(
            "No config file provided. "
            "Use --generate-example-config to create one."
        )

    config_path = Path(args.config)

    # Config file does not exist
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Run analysis
    analyzer = run_sync_analysis(**config)