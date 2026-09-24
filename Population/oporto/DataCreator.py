import argparse

def expand_census_data(input_file):
    import pandas as pd
    import geopandas as gpd
    data = gpd.read_file(input_file)

    data["N_INDIVIDUOS_5A14"] = data["N_INDIVIDUOS_5A9"] + data["N_INDIVIDUOS_10A14"]
    data["N_INDIVIDUOS_5A14_H"] = data["N_INDIVIDUOS_5A9_H"] + data["N_INDIVIDUOS_10A14_H"]
    data["N_INDIVIDUOS_5A14_M"] = data["N_INDIVIDUOS_5A9_M"] + data["N_INDIVIDUOS_10A14_M"]

    data["N_INDIVIDUOS_0A19"] = data["N_INDIVIDUOS_0A4"] + data["N_INDIVIDUOS_5A14"]+ data["N_INDIVIDUOS_15A19"]
    data["N_INDIVIDUOS_0A19_H"] = data["N_INDIVIDUOS_0A4_H"] + data["N_INDIVIDUOS_5A14_H"]+ data["N_INDIVIDUOS_15A19_H"]
    data["N_INDIVIDUOS_0A19_M"] = data["N_INDIVIDUOS_0A4_M"] + data["N_INDIVIDUOS_5A14_M"]+ data["N_INDIVIDUOS_15A19_M"]

    data["N_INDIVIDUO_ENSINCOMP_BAS"] = data["N_INDIVIDUO_ENSINCOMP_1BAS"] + data["N_INDIVIDUO_ENSINCOMP_2BAS"] + data["N_INDIVIDUO_ENSINCOMP_3BAS"]
    data["N_INDIVIDUO_ENSINCOMP_BAS_H"] = data["N_INDIVIDUO_ENSINCOMP_1BAS_H"] + data["N_INDIVIDUO_ENSINCOMP_2BAS_H"] + data["N_INDIVIDUO_ENSINCOMP_3BAS_H"]
    data["N_INDIVIDUO_ENSINCOMP_BAS_M"] = data["N_INDIVIDUO_ENSINCOMP_1BAS_M"] + data["N_INDIVIDUO_ENSINCOMP_2BAS_M"] + data["N_INDIVIDUO_ENSINCOMP_3BAS_M"]

    data["N_INDIVIDUOS_DESEMPREGADOS"] = data["N_INDIVIDUOS_DESEMPREGADOS_1EMP"] + data["N_INDIVIDUOS_DESEMPREGADOS_NOVOEMP"]
    data["N_INDIVIDUOS_DESEMPREGADOS_H"] = data["N_INDIVIDUOS_DESEMPREGADOS_1EMP_H"] + data["N_INDIVIDUOS_DESEMPREGADOS_NOVOEMP_H"]
    data["N_INDIVIDUOS_DESEMPREGADOS_M"] = data["N_INDIVIDUOS_DESEMPREGADOS_1EMP_M"] + data["N_INDIVIDUOS_DESEMPREGADOS_NOVOEMP_M"]

    data["N_INDIVIDUOS_INATIVOS"] = data["N_INDIVIDUOS_SEM_ATIVIDADE_ECONOMICA"]-(data['N_INDIVIDUOS_ESTUDANTES']+data['N_INDIVIDUOS_DOMESTICOS']+data['N_INDIVIDUOS_REFORMADOS'])
    data["N_INDIVIDUOS_INATIVOS_H"] = data["N_INDIVIDUOS_SEM_ATIVIDADE_ECONOMICA_H"]-(data['N_INDIVIDUOS_ESTUDANTES_H']+data['N_INDIVIDUOS_DOMESTICOS_H']+data['N_INDIVIDUOS_REFORMADOS_H'])
    data["N_INDIVIDUOS_INATIVOS_M"] = data["N_INDIVIDUOS_SEM_ATIVIDADE_ECONOMICA_M"]-(data['N_INDIVIDUOS_ESTUDANTES_M']+data['N_INDIVIDUOS_DOMESTICOS_M']+data['N_INDIVIDUOS_REFORMADOS_M'])

    data["N_INDIVIDUOS_NAC_PT"] = data["N_INDIVIDUOS"] - data["N_INDIVIDUOS_NAC_ESTRANGEIRA"]
    data["N_INDIVIDUOS_NAC_PT_H"] = data["N_INDIVIDUOS_H"] - data["N_INDIVIDUOS_NAC_ESTRANGEIRA_H"]
    data["N_INDIVIDUOS_NAC_PT_M"] = data["N_INDIVIDUOS_M"] - data["N_INDIVIDUOS_NAC_ESTRANGEIRA_M"]

    data["N_INDIVIDUOS_RESID_PT"] = data["N_INDIVIDUOS"] - data["N_INDIVIDUOS_RESID_FORA_PAIS"]
    data["N_INDIVIDUOS_RESID_PT_H"] = data["N_INDIVIDUOS_H"] - data["N_INDIVIDUOS_RESID_FORA_PAIS_H"]
    data["N_INDIVIDUOS_RESID_PT_M"] = data["N_INDIVIDUOS_M"] - data["N_INDIVIDUOS_RESID_FORA_PAIS_M"]

    data.to_file(f"{input_file[:-5]}_TRANSFORMED.gpkg")

def generate_places_data(input_file, output_file):
    import osmium
    import pandas as pd
    import os
    import json
    from shapely.geometry import Polygon, Point

    class POIHandler(osmium.SimpleHandler):
        def __init__(self):
            super(POIHandler, self).__init__()
            self.way_pois = []
            self.node_pois = []
        
        def node(self, n):
            self.get_poi(n, self.node_pois)
        
        def way(self, w):
            self.get_poi(w, self.way_pois)
        
        def get_poi(self, element, poi_list):
            if 'amenity' in element.tags:
                
                if isinstance(element, osmium.osm.Node):
                    x = element.location.lon
                    y = element.location.lat
                elif isinstance(element, osmium.osm.Way):
                    coords = [(node.lon, node.lat) for node in element.nodes]
                    if len(coords) < 3:
                        
                        if len(coords) == 2:
                            x = (coords[0][0] + coords[1][0]) / 2
                            y = (coords[0][1] + coords[1][1]) / 2
                        elif len(coords) == 1:
                            x, y = coords[0]
                        else:
                            return
                    else:
                        polygon = Polygon(coords)
                        centroid = polygon.centroid
                        x = centroid.x
                        y = centroid.y

        
                categories = []
                for category, possibilties in self.categories.items():
                    if element.tags['amenity'] in possibilties:
                        categories.append(category)
                    special = [p for p in possibilties if "@" in p]
                    for s in special:
                        tag, m = s.split("@")
                        key, value = m.split(":")
                        if element.tags["amenity"] == tag and value in element.tags.get(key, ""):
                            categories.append(category)
                            
                if len(categories) == 0:
                    categories.append(f"other ({element.tags['amenity']})")
                
                for cat in categories:
                    entry = (element.tags.get("name", ""), cat, y, x, element.tags.get("ref", ""))
                    poi_list.append(entry)
        
        def get_pois(self, filename, categories):
            self.categories = categories
            
            super().apply_file(filename, locations=True)
            return self.way_pois + self.node_pois

    schoolsTypes = ["school"]
    primarySchoolsTypes = ["school@name:Escola Básica", "school@name:Escola Básica e Secundária"]
    secondarySchoolTypes = ["school@name:Escola Secundária", "school@name:Escola Básica e Secundária"]
    universitiesTypes = ["university","college"]
    thirdSectorWorkPlacesTypes = [ 'animal_shelter', 'antiques', 'art', 'art_gallery', 'bank', 'batteries', 'beauty', 'charity', 'chemist', 'civic', 'clinic', 'college', 'commercial', 'community_centre', 'concert_hall', 'convenience', 'cosmetics', 'courthouse', 'coworking_space', 'dancing_school', 'deli', 'dentist', 'department_store', 'doctors', 'dojo', 'driving_school', 'drugstore', 'electronics', 'estate_agent', 'fast_food', 'fire_station', 'government', 'grocery', 'hairdresser', 'hardware', 'health_food', 'herbalist', 'hospital', 'hotel', 'ice_cream', 'industrial', 'interior_decoration', 'jewelry', 'kindergarten', 'language_school', 'library', 'mall', 'marketplace', 'museum', 'music', 'office', 'optician', 'perfumery', 'pet', 'pharmacy', 'police', 'pottery', 'research_institute', 'restaurant', 'retail', 'school', 'service', 'shelter', 'shoe_repair', 'shoes', 'sport_club', 'sports', 'sports_centre', 'stable', 'supermarket', 'veterinary']
    secondSectorWorkPlacesTypes = [ 'construction', 'factory', 'hardware', 'industrial', 'recycling', 'sewing', 'shed', 'shoe_repair','warehouse']
    firstSectorWorkPlacesTypes = ['agrarian','farm']
    shopTypes = [ 'beauty', 'books', 'clothes', 'commercial', 'department_store', 'electronics', 'fashion_accessories', 'florist', 'interior_decoration', 'mall', 'marketplace', 'perfumery', 'shoes', 'toys',]
    groceriesTypes = [ 'bakery', 'cafe', 'cheese', 'fast_food', 'grocery', 'health_food', 'herbalist', 'pastry', 'seafood', 'supermarket', 'tea']
    leisureTypes = [ 'art', 'art_gallery', 'arts_centre', 'bar', 'cafe', 'cinema', 'coffee', 'dojo', 'fast_food', 'games', 'ice_cream', 'museum', 'music', 'nightclub', 'restaurant', 'sport_club', 'sports', 'sports_centre', 'stadium', 'swimming_pool', 'tattoo', 'video_games']

    categories = {
        "primary_school":primarySchoolsTypes,
        "secondary_school":secondarySchoolTypes,
        "university":universitiesTypes,
        "workplace_1st_sec":firstSectorWorkPlacesTypes,
        "workplace_2nd_sec":secondSectorWorkPlacesTypes,
        "workplace_3rd_sec":thirdSectorWorkPlacesTypes,
        "workplace_all":firstSectorWorkPlacesTypes + secondSectorWorkPlacesTypes + thirdSectorWorkPlacesTypes,
        "shop":shopTypes,
        "groceries":groceriesTypes,
        "leisure":leisureTypes
    }

    handler = POIHandler()
    pois = handler.get_pois(input_file, categories)

    df = pd.DataFrame(pois, columns=["name", "category", "latitude", "longitude", "ref"]).drop_duplicates(subset=["ref"]).drop(columns=["ref"]).reset_index(drop=True)
    df.to_csv(output_file, index=False, float_format='%.8f')


def main():
    parser = argparse.ArgumentParser(description='Data Creator for Oporto Population IMOB')
    parser.add_argument("--expand-census", type=str, help="Expand census data")
    parser.add_argument("--generate-places", type=str, help="Generate places data")
    parser.add_argument("--output", type=str, help="Output file name")
    args = parser.parse_args()

    if args.expand_census:
        expand_census_data(args.expand_census)
    elif args.generate_places:
        generate_places_data(args.generate_places, args.output)


if __name__ == "__main__":
    main()