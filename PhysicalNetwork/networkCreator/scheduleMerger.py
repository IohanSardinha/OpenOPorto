import xml.etree.ElementTree as ET
import argparse

# Function to parse XML files and return the root element
def parse_xml(file_path):
    tree = ET.parse(file_path)
    return tree.getroot(), tree

# Function to merge two XML files
def merge_schedules(file1, file2, output_file, name1="", name2=""):
    # Parse the XML files
    root1, tree1 = parse_xml(file1)
    root2, tree2 = parse_xml(file2)

    # Merge transitStops
    transit_stops1 = root1.find("transitStops")
    transit_stops2 = root2.find("transitStops")
    if transit_stops1 is not None and transit_stops2 is not None:
        for stop in transit_stops2:
            transit_stops1.append(stop)

    # Merge minimalTransferTimes
    transfer_times1 = root1.find("minimalTransferTimes")
    transfer_times2 = root2.find("minimalTransferTimes")
    if transfer_times1 is not None and transfer_times2 is not None:
        for relation in transfer_times2:
            transfer_times1.append(relation)

    for root, name in ((root1, name1), (root2, name2)):
        if name != "":
            for departure in root.findall(".//departure"):
                departure.set("vehicleRefId", f"{name}_{departure.attrib['vehicleRefId']}".lower())

    # Merge transitLines
    for line in root2.findall("transitLine"):
        root1.append(line)

    # Add the DOCTYPE declaration
    with open(output_file, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(b'<!DOCTYPE transitSchedule SYSTEM "http://www.matsim.org/files/dtd/transitSchedule_v2.dtd">\n')
        tree1.write(f, encoding="utf-8")

# Main function to handle command line arguments
def main():
    parser = argparse.ArgumentParser(description="Merge two XML files containing transit schedule data.")
    parser.add_argument("file1", help="Path to the first XML file")
    parser.add_argument("file2", help="Path to the second XML file")
    parser.add_argument("output_file", help="Path to the output XML file")
    parser.add_argument("--name1", default="", help="Prefix for departure IDs in file1")
    parser.add_argument("--name2", default="", help="Prefix for departure IDs in file2")
    args = parser.parse_args()

    merge_schedules(args.file1, args.file2, args.output_file, args.name1, args.name2)

if __name__ == "__main__":
    main()
