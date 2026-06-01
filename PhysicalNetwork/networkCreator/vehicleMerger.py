import xml.etree.ElementTree as ET
import argparse

NS = "http://www.matsim.org/files/dtd"
NSMAP = {"m": NS}

ET.register_namespace("", NS)


def parse_xml(file_path):
    tree = ET.parse(file_path)
    return tree.getroot(), tree


def merge_vehicles(file1, file2, output_file, name1="", name2=""):
    root1, tree1 = parse_xml(file1)
    root2, _ = parse_xml(file2)

    # Collect elements
    vehicle_types = []
    vehicles = []

    for root, name in ((root1, name1), (root2, name2)):
        vehicle_types.extend(root.findall("m:vehicleType", NSMAP))
        if name != "":
            for vt in root.findall("m:vehicleType", NSMAP):
                vt.set("id", f"{name}_{vt.attrib['id']}".lower())
                
        vehicles.extend(root.findall("m:vehicle", NSMAP))
        if name != "":
            for v in root.findall("m:vehicle", NSMAP):
                v.set("id", f"{name}_{v.attrib['id']}".lower())
                v.set("type", f"{name}_{v.attrib['type']}".lower())
                

    # Remove all existing children
    for child in list(root1):
        root1.remove(child)

    # Reinsert in desired order
    for vt in vehicle_types:
        root1.append(vt)    

    for v in vehicles:
        root1.append(v)

    ET.indent(tree1, space="\t", level=0)

    tree1.write(
        output_file,
        encoding="UTF-8",
        xml_declaration=True
    )


def main():
    parser = argparse.ArgumentParser(description="Merge MATSim vehicle files")
    parser.add_argument("file1")
    parser.add_argument("file2")
    parser.add_argument("output_file")
    parser.add_argument("--name1", default="", help="Prefix for vehicle types in file1")
    parser.add_argument("--name2", default="", help="Prefix for vehicle types in file2")
    args = parser.parse_args()

    merge_vehicles(args.file1, args.file2, args.output_file, args.name1, args.name2)


if __name__ == "__main__":
    main()
