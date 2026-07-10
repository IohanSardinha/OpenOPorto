"""
Transit Synchronization Analysis GUI

Interactive interface for setting up and running transit synchronization analysis.
"""

from pywebio import input, output
import gzip
import xml.etree.ElementTree as ET
from sync_analysis import run_sync_analysis
import traceback
import json
from datetime import datetime
import sys
import io


class WebIOWriter(io.StringIO):
    """A file-like object that writes to PyWebIO output in real-time."""
    def __init__(self):
        super().__init__()
    
    def write(self, text):
        if text.strip():  # avoid empty lines
            output.put_text(text)
        return super().write(text)
    
    def flush(self):
        pass  # no-op for PyWebIO

def main():
    """Main GUI function."""
    
    # Header
    output.put_markdown("# Transit Synchronization Analysis")
    output.put_markdown("---")
    
    # Select the simulation output folder path
    output_folder = input.input(
        "Enter the path to the simulation output folder:",
        type=input.TEXT,
        value="../Simulation/output/"
    )
    
    output.put_loading()
    output.put_text("Loading schedule data... Please wait.")

    # Validate and load schedules
    try:
        f = gzip.open(output_folder + "/output_transitSchedule.xml.gz", "r")
        tree = ET.parse(f)
        root = tree.getroot()
        f.close()
    except Exception as e:
        output.put_error(f"Failed to load schedule: {str(e)}")
        return
    
    # Find transitLines
    transitLines = [
        f"{line.find('.//transportMode').text.capitalize()} {line.get('id')}" 
        for line in root.findall(".//transitLine")
    ]
    
    if not transitLines:
        output.put_error("No transit lines found in the schedule!")
        return
    
    # ===== LINE SELECTION =====
    selected_lines = []
    
    just_loaded = True

    while True:
        output.clear()

        if just_loaded:
            output.put_success("✅ Schedule loaded successfully!")
            just_loaded = False

        output.put_markdown("## Step 1: Select Transit Lines")
        output.put_markdown("Select at least 2 lines for synchronization analysis")
        output.put_markdown("---")
        
        # Show currently selected lines
        if selected_lines:
            output.put_text("Currently selected lines:")
            for line in selected_lines:
                output.put_text(f"  • {line}")
            output.put_markdown("---")
        else:
            output.put_text("No lines selected yet.")
            output.put_markdown("---")
        
        # Get available lines (not yet selected)
        available_lines = [line for line in transitLines if line not in selected_lines]
        
        if not available_lines and not selected_lines:
            output.put_error("No transit lines found in the schedule!")
            break
        
        # Build action options
        actions = []
        
        if available_lines:
            actions.append({'label': '➕ Add a line', 'value': 'add'})
        
        if selected_lines:
            actions.append({'label': '➖ Remove a line', 'value': 'remove'})
        
        if len(selected_lines) > 1:
            actions.append({'label': '✅ Done selecting lines', 'value': 'done'})
        
        if not actions:
            break
        
        # Ask user what to do next
        action = input.actions("What would you like to do?", actions)
        
        if action == 'done':
            break
        elif action == 'add':
            line = input.select("Select a transit line to add:", options=available_lines)
            selected_lines.append(line)
        elif action == 'remove':
            line = input.select("Select a transit line to remove:", options=selected_lines)
            selected_lines.remove(line)
    
    if len(selected_lines) < 2:
        output.put_error("At least 2 lines are required for synchronization analysis!")
        return
    
    # ===== EXTRACT STOPS FOR EACH SELECTED LINE =====
    line_stops = {}
    for selected_line in selected_lines:
        # Extract the line ID (remove the transport mode prefix)
        line_id = selected_line.split(' ', 1)[1]
        
        # Find the line in the XML
        line_elem = root.find(f".//transitLine[@id='{line_id}']")
        
        if line_elem is not None:
            # Get all stops from all routes in this line
            stops = set()
            for route in line_elem.findall(".//transitRoute"):
                for stop in route.findall(".//stop"):
                    stop_id = stop.get('refId')
                    stop_facility = root.find(f".//stopFacility[@id='{stop_id}']")
                    if stop_facility is not None:
                        stop_name = stop_facility.get('name')
                        if stop_id:
                            stops.add((stop_id, stop_name))
            
            line_stops[selected_line] = sorted(list(stops))
    
    # ===== CONNECTION SELECTION =====
    connections = []
    
    output.clear()
    output.put_markdown("## Step 2: Define Synchronization Connections")
    output.put_markdown("Define which stops should be synchronized between lines")
    output.put_markdown("---")
    
    while True:
        output.clear()
        
        output.put_markdown("## Step 2: Define Synchronization Connections")
        output.put_markdown("---")
        
        # Show existing connections
        if connections:
            output.put_text("Current synchronization connections:")
            for i, conn in enumerate(connections, 1):
                output.put_text(f"  {i}. {conn['line1']} [{conn['stop1']}] ↔ {conn['line2']} [{conn['stop2']}]")
            output.put_markdown("---")
        else:
            output.put_text("No connections defined yet.")
            output.put_markdown("---")
        
        # Build actions
        actions = [{'label': '➕ Add connection', 'value': 'add'}]
        
        if connections:
            actions.append({'label': '➖ Remove connection', 'value': 'remove'})
            actions.append({'label': '✅ Done', 'value': 'done'})
        
        action = input.actions("What would you like to do?", actions)
        
        if action == 'done':
            break
        elif action == 'add':
            # Select first line
            line1 = input.select("Select first transit line:", options=selected_lines)
            
            # Select stop on first line
            if line1 not in line_stops or not line_stops[line1]:
                output.put_error(f"No stops found for {line1}")
                continue
            
            stop1_display = input.select(
                f"Select stop on {line1}:", 
                options=[f"{s[0]} ({s[1]})" for s in line_stops[line1]]
            )
            # Extract stop_id from "stop_id (stop_name)" format
            stop1 = stop1_display.split(' (')[0]
            
            # Select second line (exclude the first line for meaningful connections)
            other_lines = [line for line in selected_lines if line != line1]
            if not other_lines:
                output.put_error("Need at least 2 different lines to create a connection!")
                continue
            
            line2 = input.select("Select second transit line:", options=other_lines)
            
            # Select stop on second line
            if line2 not in line_stops or not line_stops[line2]:
                output.put_error(f"No stops found for {line2}")
                continue
            
            stop2_display = input.select(
                f"Select stop on {line2}:", 
                options=[f"{s[0]} ({s[1]})" for s in line_stops[line2]]
            )
            # Extract stop_id from "stop_id (stop_name)" format
            stop2 = stop2_display.split(' (')[0]
            
            # Add connection
            connections.append({
                'line1': line1,
                'stop1': stop1,
                'line2': line2,
                'stop2': stop2
            })
            
            output.put_success(f"Added connection: {line1} [{stop1}] ↔ {line2} [{stop2}]")
            
        elif action == 'remove':
            # Create readable options for removal
            connection_options = [
                f"{conn['line1']} [{conn['stop1']}] ↔ {conn['line2']} [{conn['stop2']}]"
                for conn in connections
            ]
            
            conn_to_remove = input.select("Select connection to remove:", options=connection_options)
            idx = connection_options.index(conn_to_remove)
            connections.pop(idx)
    
    if not connections:
        output.put_error("At least one connection is required!")
        return
    
    # ===== TRANSFORM DATA INTO REQUIRED FORMAT =====
    # Extract just the line IDs (without transport mode prefix)
    syncLines = [line.split(' ', 1)[1] for line in selected_lines]
    
    # Build syncNodes structure: list of lists of tuples (line_id, stop_id)
    syncNodes = []
    for conn in connections:
        line1_id = conn['line1'].split(' ', 1)[1]
        line2_id = conn['line2'].split(' ', 1)[1]
        
        sync_pair = [
            (line1_id, conn['stop1']),
            (line2_id, conn['stop2'])
        ]
        syncNodes.append(sync_pair)
    
    # ===== FINAL CONFIGURATION SUMMARY =====
    output.clear()
    output.put_markdown("# Configuration Summary")
    output.put_markdown("---")
    
    output.put_text("Selected Transit Lines:")
    for line in selected_lines:
        output.put_text(f"  • {line}")
    
    output.put_markdown("---")
    output.put_text("Synchronization Connections:")
    if connections:
        for i, conn in enumerate(connections, 1):
            output.put_text(f"  {i}. {conn['line1']} Stop [{conn['stop1']}] ↔ {conn['line2']} Stop [{conn['stop2']}]")
    
    output.put_markdown("---")
    output.put_text("Generated Data Structures:")
    output.put_code(f"syncLines = {syncLines}", language="python")
    output.put_code(f"syncNodes = {syncNodes}", language="python")
    
    output.put_markdown("---")
    
    # Ask for results folder
    results_folder = input.input(
        "Enter the path for results output:",
        type=input.TEXT,
        value="results"
    )
    
    sync_analysis_config = {
        "simulation_folder":output_folder,
        "sync_lines":syncLines,
        "sync_nodes":syncNodes,
        "output_folder":results_folder
    }

    with open(f"sync_config_Lines-{'-'.join(syncLines)}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.json", "w") as f:
        json.dump(sync_analysis_config, f, indent=4)
        output.put_success(f"✅ Configuration saved to JSON file: {f.name}")

    # Confirm before running
    output.put_markdown("---")
    ready = input.actions(
        "Ready to run the synchronization analysis?",
        [
            {'label': '▶️ Run Analysis', 'value': 'run'},
            {'label': '❌ Cancel', 'value': 'cancel'}
        ]
    )
    
    if ready == 'cancel':
        output.put_warning("Analysis cancelled.")
        return
    
    # ===== RUN THE ANALYSIS =====
    output.clear()
    output.put_markdown("# Running Synchronization Analysis")
    output.put_markdown("---")
    
    try:
        with output.put_loading():
            output.put_text("This may take a few minutes.")
            
            # Redirect stdout temporarily
            old_stdout = sys.stdout
            sys.stdout = WebIOWriter()
            
            results_path = run_sync_analysis(**sync_analysis_config)

            sys.stdout = old_stdout  # restore stdout
        
        output.clear()
        output.put_success("✅ Analysis completed successfully!")
        output.put_markdown("---")
        
        output.put_text(f"Results have been saved to: {results_path}")
        output.put_markdown("### Generated Files:")
        output.put_text("  • lines.csv")
        output.put_text("  • sync_nodes.csv")
        output.put_text("  • sync_nodes_matrix.csv")
        output.put_text("  • sync_nodes_summary.csv")
        output.put_text("  • number_of_transfers.csv")
        output.put_text("  • schedule_deviations.csv")
        output.put_text("  • travel_times.csv")
        output.put_text("  • headways.csv")
        output.put_text("  • complete/ (detailed versions)")
        
    except Exception as e:
        output.put_error(f"Analysis failed: {str(e)}")
        output.put_text("Detailed error:")
        output.put_code(traceback.format_exc(), language="text")


if __name__ == "__main__":
    main()