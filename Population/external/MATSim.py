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


        activity_open = lambda person,i: f"""\t\t\t<activity type="{person[f'leg_{i}_activity']}" x="{person[f'leg_{i}_x']}" y="{person[f'leg_{i}_y']}" end_time="{person[f'leg_{i}_arrival']}">\n"""
        activity_close = "\t\t\t</activity>\n"
        leg_open = lambda person,i: f"\t\t\t<leg mode=\"{person[f'leg_{i}_mode']}\">\n"
        leg_close = "\t\t\t</leg>\n"


        for i, person in self.population.iterrows():
                    
            trips_xml = "".join(
                ["".join([activity_open(person, i), activity_close, leg_open(person, i), leg_close])
                for i in range(person["leg_count"])]+[activity_open({"leg_0_activity": person[f"leg_0_activity"], "leg_0_x": person[f"leg_0_x"], "leg_0_y": person[f"leg_0_y"], "leg_0_arrival": "23:59:59"}, 0), activity_close]
            )
            
            if self.id_builder is None:
                person_id = "_".join(
                    self.__clean_string(str(v))
                    for v in [person[c] for c in self.population.columns if not "leg_" in c]
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