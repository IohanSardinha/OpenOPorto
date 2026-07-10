from dataclasses import dataclass, field
import pandas as pd
from .TripCleaner import TripCleaner
from .ActivityTypes import IMobActivity

@dataclass
class HouseHold:
    id: str
    dtcc: str
    zone: str
    weight: float
    members:list = field(default_factory=lambda:{})
    income: str = None
    amount_vehicles: int = None
    fuel_expense:str = None
    parking_expense:str = None
    toll_expense:str = None
    transport_expense:str = None

@dataclass
class Individual:
    id: str
    household: HouseHold
    relationToHousehold: str
    gender: str
    ageGroup: str
    educationLvl: str
    EconomicSituation: str
    hasLicense: bool
    drivingFrequency: str
    workingSpaceType: str
    madeTrip: bool
    mobilityPass: str = None
    trip: list = field(default_factory=lambda:[])

@dataclass
class Leg:
    mode:str
    duration:int
    distance:int
    operator:str
    trasportPass:str
    passangers:int

@dataclass
class Trip:
    id:str
    individual:Individual
    motivation:str
    mode:str
    type:str
    duration:int
    distance:int
    dayOfWeek:str
    weekDay:bool
    departureTime:int
    departureZone:str
    departureLocation:str
    arrivalTime:int
    arrivalZone:str
    arrivalLocation:str
    legs:list

class IMobProcesser:

    @staticmethod
    def __readHouseholds(file):
        householdsDF = pd.read_csv(file, sep=";")

        households = {
            row.Id_aloj_1: HouseHold(row.Id_aloj_1, row.DTCC_aloj, row.Zona_aloj, row.PESOFIN)
            for row in householdsDF.itertuples(index=False)
        }
        return households

    @staticmethod
    def __readExpenses(file, households):
        expenses = pd.read_csv(file, sep=";")

        for row in expenses.itertuples(index=False):
            h = households.get(row.Id_aloj_1)
            h.fuel_expense = row.Desp_Comb_Esc_Dsg
            h.parking_expense = row.Desp_Esta_Esc_Dsg
            h.toll_expense = row.Desp_Port_Esc_Dsg
            h.transport_expense = row.Desp_Tp_Esc_Dsg
    
    @staticmethod
    def __readVehicles(file, households):
        vehicles = pd.read_csv(file, sep=";")

        vehicle_counts = vehicles["Id_aloj_1"].value_counts()

        for id_, h in households.items():
            #Add to household just the amount of cars(ignoring some other info like size, fuel type and year of car)
            h.amount_vehicles = vehicle_counts.get(id_, 0)

    @staticmethod
    def __readIncomes(file, households):
        incomes = pd.read_csv(file, sep=";")

        for row in incomes.itertuples(index=False):
            h = households.get(row.Id_aloj_1)
            h.income = row.Rendimento_Dsg

    @staticmethod
    def __readIndividuals(file, households):
        individualsData = pd.read_csv(file, sep=";")
        individuals = {}

        for row in individualsData.itertuples(index=False):
            household = households[row.Id_aloj_1]
            license = ( row.Carta_C1 == 1 or row.Carta_C2 == 1 or row.Carta_C3 == 1 or row.Carta_C4 == 2 )
            i = Individual(
                f"{row.Id_aloj_1}.{row.N_Individuo}",
                household,
                row.Parentesco_Dsg,
                row.Sexo_Dsg,
                row.Idade_Cod_Dsg,
                row.Nivel_Instr_Cod_Dsg,
                row.Cond_Trab_Cod_Dsg,
                license,
                row.Conduz_Dsg,
                row.Ltrab_Tipo_Dsg,
                row.D0100_Dsg == "Sim"
            )

            household.members[i.id] = i
            individuals[i.id] = i
        
        return individuals

    @staticmethod
    def __readPasses(file, individuals):
        passes = pd.read_csv(file, sep=";")

        for row in passes.itertuples(index=False):
            i = individuals.get(f"{row.Id_aloj_1}.{row.N_Individuo}")
            i.mobilityPass = row.Passe_Operador1_Dsg

    @staticmethod
    def __readTrips(file, individuals):
        tripsData = pd.read_csv(file, sep=";", low_memory=False)
        tripsData = tripsData[tripsData["Hora_chegada"].notna() & tripsData["Hora_partida"].notna()]
        tripsData["Hora_partida"] = pd.to_datetime(tripsData["Hora_partida"], errors="coerce")
        tripsData["Hora_chegada"] = pd.to_datetime(tripsData["Hora_chegada"], errors="coerce")
        tripsData["Duracao"] = pd.to_datetime(tripsData["Duracao"], errors="coerce")
        tripsData["mode"] = (
            tripsData["TI"].eq("S").map({True: "car", False: ""})
            + tripsData["TP"].eq("S").map({True: "+pt", False: ""})
        ).str.strip("+")
        tripsData["mode"] = tripsData["mode"].replace("","walk")
        trips = {}

        for row in tripsData.itertuples(index=False):
            individual_id = f"{row.Id_aloj_1}.{row.N_Individuo}"
            trip_id = f"{individual_id}.{row.N_Desloc}"

            legs = []
            for i in range(1, 6):
                transp = getattr(row, f"Et{i}_transp")
                if isinstance(transp, str):
                    legs.append(Leg(
                        transp,
                        getattr(row, f"Et{i}_Duracao"),
                        getattr(row, f"Et{i}_Distancia"),
                        getattr(row, f"ET{i}_Operador"),
                        getattr(row, f"ET{i}_Titulo_transp"),
                        getattr(row, f"ET_{i}_passageiros")
                    ))

            trip = Trip(
                trip_id,
                individuals[individual_id],
                row.D0500_Dsg,
                row.mode,
                row.Tipo,
                row.Duracao.time() if pd.notna(row.Duracao) else None,
                row.Distancia,
                row.Dia_da_semana,
                row.Dia_util == 1,
                row.Hora_partida.time() if pd.notna(row.Hora_partida) else None,
                row.Zona_or,
                row.DTCC_or,
                row.Hora_chegada.time() if pd.notna(row.Hora_chegada) else None,
                row.Zona_de,
                row.DTCC_de,
                legs
            )

            individuals[individual_id].trip.append(trip)
            trips[trip.id] = trip

        return trips

    @staticmethod
    def __remapActivity(A):
        if A == 'Ir para o trabalho':
            return IMobActivity.WORK
        if A == 'Levar/buscar/acompanhar familiares ou amigos (crianças à escola, etc)':
            return IMobActivity.TAKE_SOMEONE_SOMEWHERE
        if A == 'Regressar a casa':
            return IMobActivity.HOME
        if A == 'Fazer compras (supermercado, mercearia, utilidades, etc)':
            return IMobActivity.GROCERIES
        if A == 'Ir para a escola ou atividades escolares':
            return IMobActivity.SCHOOL
        if A == 'Fazer percurso pedonal (início e fim no mesmo local), jogging, passear o cão, etc. (com pelo menos 200 metros)':
            return IMobActivity.AROUND_THE_BLOCK
        if A == 'Praticar atividades ao ar livre (desporto ou lazer) ou em ginásio ou pavilhão':
            return IMobActivity.WORKOUT
        if A == 'Visitar familiares ou amigos':
            return IMobActivity.VISIT_FRIEND_FAMILY
        if A == 'Ir a restaurante, café, bar, discoteca, etc.':
            return IMobActivity.EAT_OUT
        if A == 'Outra atividade':
            return IMobActivity.OTHER
        if A == 'Tratar de assuntos profissionais':
            return IMobActivity.WORK
        if A == 'Assistir a eventos desportivos ou culturais (cinema, teatro, concerto, futebol, etc.)':
            return IMobActivity.LEASURE_SPORT_OR_CULURAL
        if A == 'Tratar de assuntos pessoais (ir ao banco, lavandaria, cabeleireiro, levar ou buscar coisas pessoais, etc)':
            return IMobActivity.PERSONAL_ISSUES
        if A == 'Outras atividades de lazer, entretenimento ou turismo':
            return IMobActivity.LEASURE_OTHER
        if A == 'Ir a consulta, tratamentos, exames médicos e similares':
            return IMobActivity.DOCTOR
        if A == 'Realizar atividade em grupo ou em contexto coletivo (em associações, comícios, igrejas, voluntariado, ...)':
            return IMobActivity.LEASURE_COLLECTIVE

    @staticmethod
    def __toGenericFormat(individuals):
        #Can be improved
        getTripType = lambda i: " - ".join(list(set(" - ".join([t.type for t in i.trip]).split(" - "))))
        return {
                id:{
                        "attributes":{
                            "gender":i.gender,
                            "ageGroup":i.ageGroup,
                            "educationLvl":i.educationLvl,
                            "economicSituation":i.EconomicSituation,
                        },
                        "tripDesc":{
                            "type":getTripType(i),
                            "weekday":i.trip[0].weekDay
                        },
                        "legs":[{
                                "activity":IMobProcesser.__remapActivity(t.motivation),
                                "distance":t.distance,
                                "mode":t.mode,
                                "departure":t.departureTime,
                                "arrival":t.arrivalTime
                            }for t in i.trip]
                    } for id, i in individuals.items() if i.madeTrip and len(i.trip) > 0
                }

    @staticmethod
    def __read_files(householdsFile, expensesFile,vehiclesFile,incomesFile,individualsFile,passesFile,tripsFile, fix_trips=True):
        households = IMobProcesser.__readHouseholds(householdsFile)
        IMobProcesser.__readExpenses(expensesFile, households)
        IMobProcesser.__readVehicles(vehiclesFile, households)
        IMobProcesser.__readIncomes(incomesFile, households)
        individuals = IMobProcesser.__readIndividuals(individualsFile, households)
        IMobProcesser.__readPasses(passesFile, individuals)
        IMobProcesser.__readTrips(tripsFile, individuals)
        
        genericFormat = IMobProcesser.__toGenericFormat(individuals)

        if fix_trips:
            for id in genericFormat.keys():
                genericFormat[id]["legs"] = TripCleaner.fix_trip(genericFormat[id])
        
        return genericFormat
    

    @staticmethod
    def read(**kwargs):
        if "folder" in kwargs:
            folder = kwargs["folder"]
            return IMobProcesser.__read_files(
                kwargs.get("householdsFile", f"{folder}/TBL_alojamento_AMP.csv"),
                kwargs.get("expensesFile", f"{folder}/TBL_alojamento_despesa_AMP.csv"),
                kwargs.get("vehiclesFile", f"{folder}/TBL_alojamento_veiculos_AMP.csv"),
                kwargs.get("incomesFile", f"{folder}/TBL_alojamento_rendimentos_AMP.csv"),
                kwargs.get("individualsFile", f"{folder}/TBL_individuos_AMP.csv"),
                kwargs.get("passesFile", f"{folder}/TBL_tipo_de_passe_AMP.csv"),
                kwargs.get("tripsFile", f"{folder}/TBL_viagens_AMP.csv"),
                kwargs.get("fix_trips", True)
            )
        else:
            return IMobProcesser.__read_files(
                kwargs["householdsFile"],
                kwargs["expensesFile"],
                kwargs["vehiclesFile"],
                kwargs["incomesFile"],
                kwargs["individualsFile"],
                kwargs["passesFile"],
                kwargs["tripsFile"],
                kwargs.get("fix_trips", True)
            )