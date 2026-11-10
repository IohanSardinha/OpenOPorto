import argparse

def expand_census_data(input_file):
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

    vars = {"ageGroup":['N_INDIVIDUOS_0A4', 'N_INDIVIDUOS_5A9', 'N_INDIVIDUOS_10A14', 'N_INDIVIDUOS_15A19', 'N_INDIVIDUOS_20A24', 'N_INDIVIDUOS_25A29', 'N_INDIVIDUOS_30A34', 'N_INDIVIDUOS_35A39', 'N_INDIVIDUOS_40A44', 'N_INDIVIDUOS_45A49', 'N_INDIVIDUOS_50A54', 'N_INDIVIDUOS_55A59', 'N_INDIVIDUOS_60A64', 'N_INDIVIDUOS_65A69', 'N_INDIVIDUOS_70A74', 'N_INDIVIDUOS_75_OU_MAIS'],
        "gender": ["H", "M"]}
    section_col = "DTMNFRSEC21"
    section_age_gender = data.melt(id_vars=[section_col], value_vars=[f"{age_group}_{gender}" for age_group in vars["ageGroup"] for gender in vars["gender"]], var_name="age_gender", value_name="count")
    section_age_gender[["ageGroup", "gender"]] = section_age_gender["age_gender"].str.extract(r"(.+)_(H|M)$")
    section_age_gender = section_age_gender.groupby([section_col, "gender", "ageGroup"], as_index=False)["count"].sum().rename(columns={section_col: "section"})
    section_age_gender["ageGroup"] = section_age_gender["ageGroup"].str.replace("N_INDIVIDUOS_", "").str.replace("_OU_MAIS", "+").str.replace("A", " to ")
    section_age_gender["gender"] = section_age_gender["gender"].map({"H": "Male", "M": "Female"})
    section_age_gender["section"] = section_age_gender["section"].astype(str)

    section_age_gender.to_csv(f"{input_file[:-5]}_SECTION_AGE_GENDER.csv", index=False)

def preprocess_ipums_data(input_file, output_file):
    import pandas as pd
    from ipumspy import readers

    ddi = readers.read_ipums_ddi(f"{input_file}.xml")
    df = readers.read_microdata_chunked(ddi, f"{input_file}.dat",chunksize=10000)

    chunk = next(df, None)
    part = chunk[chunk["GEO1_PT2011"] == 114]
    while chunk is not None:
        part = pd.concat([part, chunk[chunk["GEO1_PT2011"] == 114]])
        chunk = next(df, None)
    porto_df = part

    invert_dict = lambda d: {v: k for k, v in d.items()}

    persons = porto_df[["AGE2", "SEX", "EDUCPT", "EEMPSTAT","SCHOOL", "TRNWRK","CITIZEN"]].copy()
    for col in persons.columns:
        mapper = invert_dict(ddi.get_variable_info(col).codes)
        persons[col] = persons[col].map(mapper)

    persons["EEMPSTAT"] = persons["EEMPSTAT"].replace("Unemployed, never worked before","Unemployed")
    persons.loc[persons["AGE2"].isin(["0 to 4", "5 to 9", "10 to 14", "15 to 19"]), "EEMPSTAT"] = "Students"
    persons.loc[persons["AGE2"].isin(["0 to 4", "5 to 9", "10 to 14", "15 to 19"]), "SCHOOL"] = "Yes"
    persons["TRNWRK"] = persons["TRNWRK"].replace({"Auto (driver)":"Car",
                                                "Auto (passenger)":"Car", 
                                                "Bus or trolley bus" : "Public Transport", 
                                                "Railroad or train": "Public Transport", 
                                                "Subway or elevated train": "Public Transport", 
                                                "Ferryboat": "Public Transport",
                                                "Special transportation (company coach, school bus, etc.)": "Car",
                                                "Motorcycle, moped, or scooter":"Car",
                                                "Bicycle":"Walking",
                                                "Unknown/missing": "Other",
                                                "Other, not elsewhere classified":"Other",
                                                "NIU (not in universe)":"Other"
                                                })

    persons["EDUCPT"] = persons["EDUCPT"].replace({"Basic education, grades 1-4":"1 Basic",
                                                "Basic education, grades 5-6":"2 Basic",
                                                "Basic education, grades 7-9":"3 Basic",
                                                "Basic education, grades 10-12":"Highschool",
                                                "Specialized technical school (level 4)":"Highschool",
                                                "Less than primary, illiterate": "No Education",
                                                "Less than primary, literate": "No Education",
                                                "Bachelor's degree":"Undergraduate+",
                                                "Professional degree (licenciature)":"Undergraduate+",
                                                "Master's and PhD":"Undergraduate+"})

    persons["CITIZEN"] = persons["CITIZEN"].replace({'Citizen, not specified':'Portuguese', 'Not a citizen':'Non-Portuguese'})

    #For now, all SCHOOL = Yes are students
    persons.loc[persons["SCHOOL"] == "Yes", "EEMPSTAT"] = "Students"
    persons.drop(columns=["SCHOOL"], inplace=True)
    persons = persons.rename(columns={"AGE2":"ageGroup", "SEX":"gender", 
                                      "EDUCPT":"educationLvl", 
                                      "CITIZEN":"citizenship",
                                      "EEMPSTAT":"economicSituation", 
                                      "TRNWRK":"transportationMode"})
    
    persons.to_csv(output_file, index=False)

def generate_places_data(input_file, output_file):
    import osmium
    import pandas as pd
    from shapely.geometry import Polygon

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
    parser.add_argument("--preprocess-ipums", type=str, help="Preprocess IPUMS data")
    parser.add_argument("--generate-places", type=str, help="Generate places data")
    parser.add_argument("--output", type=str, help="Output file name")
    args = parser.parse_args()

    if args.expand_census:
        expand_census_data(args.expand_census)
    elif args.preprocess_ipums:
        preprocess_ipums_data(args.preprocess_ipums, args.output)
    elif args.generate_places:
        generate_places_data(args.generate_places, args.output)


if __name__ == "__main__":
    main()