import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

def _parse_xml(file_path):
    tree = ET.parse(file_path)
    return tree.getroot(), tree

def extract_vehicle_info(vehicle_file):
    NS = "http://www.matsim.org/files/dtd"
    NSMAP = {"m": NS}

    ET.register_namespace("", NS)

    root, tree = _parse_xml(vehicle_file)
    
    vehicle_types = []

    vehicle_types.extend(map(lambda v: v.find("m:networkMode", NSMAP).get("networkMode"), root.findall("m:vehicleType", NSMAP)))
    
    return vehicle_types

def extract_plan_info(plan_file):

    root, tree = _parse_xml(plan_file)
    
    df = "%H:%M:%S"
    activity_type_times = {}
    earliest_time_all = datetime.strptime("23:59:59", df)
    latest_time_all = datetime.strptime("00:00:00", df)
    modes = set()


    for person in root.findall("person"):
        activities = person.find("plan").findall("activity")
        for a1, a2 in zip(activities[-1:]+activities[:-1], activities):
            act_type = a2.get("type")
            act_start_time = datetime.strptime(a1.get("end_time"), df)
            act_end_time = datetime.strptime(a2.get("end_time"), df)
            if act_type not in activity_type_times:
                activity_type_times[act_type] = {"avg": [],"earliest": datetime.strptime("00:00:00",df),"latest": datetime.strptime("23:59:59",df)}       
            dt1 = (act_end_time - act_start_time).total_seconds()
            dt2 = (act_start_time - act_end_time).total_seconds()
            dt = dt1 if dt1 >=0 else dt2
            activity_type_times[act_type]["avg"].append(dt)
            if act_start_time < activity_type_times[act_type]["earliest"]:
                activity_type_times[act_type]["earliest"] = act_start_time
            if act_end_time > activity_type_times[act_type]["latest"]:
                activity_type_times[act_type]["latest"] = act_end_time
            if act_start_time < earliest_time_all:
                earliest_time_all = act_start_time
            if act_end_time > latest_time_all:
                latest_time_all = act_end_time
        
        for leg in person.find("plan").findall("leg"):
            modes.add(leg.get("mode"))

    for k in activity_type_times.keys():
        activity_type_times[k]["avg"] = str(timedelta(seconds=sum(activity_type_times[k]["avg"])/len(activity_type_times[k]["avg"]))).split(".")[0]
        activity_type_times[k]["earliest"] = str(activity_type_times[k]["earliest"].time())
        activity_type_times[k]["latest"] = str(activity_type_times[k]["latest"].time())

    return str(earliest_time_all.time()), str(latest_time_all.time()), activity_type_times, modes