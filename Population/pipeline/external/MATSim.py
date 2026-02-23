import unicodedata
import re

class MATSimPopulationExporter():
    def __init__(self, population, id_builder=None, format="PYTHON"):
        if format == "JSON":
            self.population = self.from_JSON(population).population
        else:
            self.population = population
        self.id_builder = id_builder
    
    def from_JSON(self, population):
        import json
        if isinstance(population, str):
            with open(population, "r") as f:
                self.population = json.load(f)
        else:
            self.population = population
        return self

    def __clean_string(self, s):
        # 1. Normalize accents
        s = unicodedata.normalize('NFKD', s)
        s = s.encode('ascii', 'ignore').decode('ascii')

        # 2. CamelCase → snake_case
        s = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s)
        s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)

        # 3. Replace non-alphanumeric with underscore
        s = re.sub(r'[^a-zA-Z0-9]+', '_', s)

        # 4. Lowercase and strip extra underscores
        return s.lower().strip('_')

    def __to_XML(self):
        parts = []

        parts.append('<?xml version="1.0" encoding="utf-8"?>\n')
        parts.append('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">\n')
        parts.append("<population>\n")

        activity_open = lambda leg: f"""\t\t\t<activity type="{leg['activity']}" x="{leg['x']}" y="{leg['y']}" end_time="{leg['arrival']}">\n"""
        activity_close = "\t\t\t</activity>\n"
        leg_open = lambda leg: f"\t\t\t<leg mode=\"{leg['mode']}\">\n"
        leg_close = "\t\t\t</leg>\n"


        for i, person in enumerate(self.population):

            last_activity = person["trips"][0].copy()
            last_activity["arrival"] = "23:59:59"
            
            trips_xml = "".join(
                ["".join([activity_open(leg), activity_close, leg_open(leg), leg_close])
                for leg in person["trips"]]+[activity_open(last_activity), activity_close]
            )
            
            if self.id_builder is None:
                person_id = "_".join(
                    self.__clean_string(str(v))
                    for v in person["attributes"].values()
                )
            else:
                person_id = self.id_builder(person)

            parts.append(f'\t<person id="person_{i}_{person_id}">\n')
            parts.append('\t\t<plan selected="yes">\n')
            parts.append(trips_xml)
            parts.append('\t\t</plan>\n')
            parts.append('\t</person>\n')

        parts.append("</population>")

        return "".join(parts)


    def as_XML(self):
        self.xml = self.__to_XML()
        return self
    
    def to_XML(self):
        self.xml = self.__to_XML()
        return self.xml

    def export(self, filepath):
        with open(filepath, "w") as f:
            f.write(self.xml)