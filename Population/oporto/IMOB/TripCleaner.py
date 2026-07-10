import datetime
import random
from .ActivityTypes import IMobActivity

class TripCleaner:

    @staticmethod
    def __twoFollowed(_list, target):
        for idx, element in enumerate(_list[0:-1]):
            if element == _list[idx+1] == target:
                return True
        return False

    @staticmethod
    def map_economic_situation(econ):
        if econ == 'Empregado(a)':
            return "Worker"
        elif econ == 'Estudante, Reformado(a), Ocupa-se principalmente de tarefas domésticas, Incapacitado(a) permanente ou Outra situação de inatividade':
            return "StayAtHomeStudentOrReformed"
        else:
            return "Other"

    @staticmethod
    def __fix_followed_homes(trip):
        #Two or more homes followed
        while TripCleaner.__twoFollowed([p["activity"] for p in trip], IMobActivity.HOME):
            start = None
            for i,p in enumerate(trip):
                if p["activity"] == IMobActivity.HOME and start == None:
                    start = i
                if (start != None and p["activity"] != IMobActivity.HOME) or ((i == len(trip)-1) and p["activity"] == IMobActivity.HOME):
                    end = i if (i == len(trip)-1) else i-1
                    if end > start:

                        distance = sum([float(p["distance"]) for p in trip[start:end]])
                        allPT  = all([p["mode"] == "pt" for p in trip[start:end]])
                        allCar = all([p["mode"] == "car" for p in trip[start:end]])
                        mode = "pt" if allPT else ("car" if allCar else "car+pt")

                        newTrip = {"activity": IMobActivity.HOME,
                                "distance": distance,
                                "mode": mode,
                                "departure": trip[start]["departure"], 
                                "arrival": trip[end]["arrival"]
                                }
                        trip = trip[:start] + [newTrip] + trip[end+1:]
                        break
                    else:
                        start = None
        
        if len(trip) > 2 and trip[0]["activity"] == IMobActivity.HOME == trip[-1]["activity"]:
            trip = trip[1:]

        return trip

    @staticmethod
    def __fix_trip_single_element(attributes, trip):
        #No Home
        if len(trip) == 1 and not trip[0]["activity"] == IMobActivity.HOME:
            if trip[0]["activity"] == IMobActivity.WORK:
                dt2  = datetime.timedelta(hours=8)
            elif trip[0]["activity"] == IMobActivity.SCHOOL:
                dt2 = datetime.timedelta(hours=5)
            else:
                dt2 = datetime.timedelta(hours=random.randint(0,7), minutes=random.randint(1,59))
            
            ts = datetime.datetime.combine(datetime.date.today(), trip[0]["departure"])
            tt = datetime.datetime.combine(datetime.date.today(), trip[0]["arrival"])

            dt1 = tt-ts

            departure = (tt+dt2)
            departure = datetime.time(departure.hour, departure.minute, departure.second)

            arrival = (tt+dt2+dt1)
            arrival = datetime.time(arrival.hour, arrival.minute, arrival.second)

            trip.append({
                "activity":IMobActivity.HOME,
                "distance":trip[0]["distance"],
                "mode":trip[0]["mode"],
                "departure":departure,
                "arrival":arrival
            })
        #Just Home
        elif len(trip) == 1 and trip[0]["activity"] == IMobActivity.HOME:
            if attributes["ageGroup"] == "0-14":
                act = IMobActivity.SCHOOL
                dt2 = datetime.timedelta(hours=5)
            elif attributes["ageGroup"] == "15-24" or attributes["ageGroup"] == "25-44":
                if TripCleaner.map_economic_situation(attributes["economicSituation"]) == "StayAtHomeStudentOrReformed":
                    act = IMobActivity.SCHOOL
                    dt2 = datetime.timedelta(hours=5)
                elif TripCleaner.map_economic_situation(attributes["economicSituation"]) == "Worker":
                    act = IMobActivity.WORK
                    dt2 = datetime.timedelta(hours=8)
                else:
                    act = random.sample([IMobActivity.TAKE_SOMEONE_SOMEWHERE,IMobActivity.GROCERIES,IMobActivity.AROUND_THE_BLOCK,IMobActivity.WORKOUT,IMobActivity.VISIT_FRIEND_FAMILY,IMobActivity.EAT_OUT,IMobActivity.OTHER,IMobActivity.LEASURE_SPORT_OR_CULURAL,IMobActivity.PERSONAL_ISSUES,IMobActivity.LEASURE_OTHER,IMobActivity.DOCTOR,IMobActivity.LEASURE_COLLECTIVE],1)[0]
                    dt2 = datetime.timedelta(hours=random.randint(0,7), minutes=random.randint(1,59))
            else:
                if TripCleaner.map_economic_situation(attributes["economicSituation"]) == "Worker":
                    act = IMobActivity.WORK
                    dt2 = datetime.timedelta(hours=random.sample([4,8],1)[0])
                else:
                    act = random.sample([IMobActivity.TAKE_SOMEONE_SOMEWHERE,IMobActivity.GROCERIES,IMobActivity.AROUND_THE_BLOCK,IMobActivity.WORKOUT,IMobActivity.VISIT_FRIEND_FAMILY,IMobActivity.EAT_OUT,IMobActivity.OTHER,IMobActivity.LEASURE_SPORT_OR_CULURAL,IMobActivity.PERSONAL_ISSUES,IMobActivity.LEASURE_OTHER,IMobActivity.DOCTOR,IMobActivity.LEASURE_COLLECTIVE],1)[0]
                    dt2 = datetime.timedelta(hours=random.randint(0,7), minutes=random.randint(1,59))
            
            ts = datetime.datetime.combine(datetime.date.today(), trip[0]["departure"])
            tt = datetime.datetime.combine(datetime.date.today(), trip[0]["arrival"])

            dt1 = tt-ts

            departure = (tt+dt2)
            departure = datetime.time(departure.hour, departure.minute, departure.second)

            arrival = (tt+dt2+dt1)
            arrival = datetime.time(arrival.hour, arrival.minute, arrival.second)

            trip.append({
                "activity":act,
                "distance":trip[0]["distance"],
                "mode":trip[0]["mode"],
                "departure":departure, 
                "arrival":arrival
            })

        return trip

    @staticmethod
    def __fix_missing_home(trip):
        # MUST CHANGE WHEN UPDATING TO TIME BASED DISTANCES
        #No Home
        if not IMobActivity.HOME in [p["activity"] for p in trip]:
            distance = sum([float(p["distance"]) for p in trip])/len(trip)
            deltas = [datetime.datetime.combine(datetime.date.today(), p["arrival"]) - datetime.datetime.combine(datetime.date.today(), p["departure"]) for p in trip]
            dt1 = datetime.timedelta(0)
            for d in deltas: dt1 += d
            dt1 = dt1/len(trip)

            ts = datetime.datetime.combine(datetime.date.today(), trip[0]["departure"])+datetime.timedelta(days=1)
            te = datetime.datetime.combine(datetime.date.today(), trip[-1]["arrival"])
            dt3 = ts-te

            mode = random.sample([p["mode"] for p in trip],1)[0]

            if trip[-1]["activity"] == IMobActivity.SCHOOL: dt2 = min(datetime.timedelta(hours=5), datetime.timedelta(seconds=dt3.seconds))
            elif trip[-1]["activity"] == IMobActivity.WORK: dt2 = min(datetime.timedelta(hours=8), datetime.timedelta(seconds=dt3.seconds))
            else: dt2 = datetime.timedelta(seconds=random.randint(1800,dt3.seconds))
            
            nts = te + dt2
            nte = nts + dt1

            #Should have a different name indicating artificiality
            np = {"activity":IMobActivity.HOME,
                "distance":distance,
                "mode":mode,
                "departure":datetime.time(nts.hour, nts.minute, nts.second),
                "arrival":datetime.time(nte.hour, nte.minute, nte.second)
                }
            if nts.day == te.day:
                trip.append(np)
            else:
                trip.insert(0, np)
        return trip

    @staticmethod
    def __fix_weird_distances(trip):
        # MUST CHANGE WHEN UPDATING TO TIME BASED DISTANCES
        if len(trip) == 2 and (float(trip[0]["distance"]) - float(trip[1]["distance"]) > 1000):
            u = (float(trip[0]["distance"])+float(trip[1]["distance"]))/2
            trip[0] = {"activity":trip[0]["activity"],
                    "distance":u,
                    "mode":trip[0]["mode"],
                    "departure":trip[0]["departure"],
                    "arrival":trip[0]["arrival"]
                    }
            trip[1] = {"activity":trip[1]["activity"],
                    "distance":u,
                    "mode":trip[1]["mode"],
                    "departure":trip[1]["departure"],
                    "arrival":trip[1]["arrival"]
                    }

        for leg in trip:
            if float(leg["distance"]) > 22222:
                trip = [{"activity":p["activity"],
                        "distance":min(float(p["distance"]),22222),
                        "mode":p["mode"],
                        "departure":p["departure"],
                        "arrival":p["arrival"]
                        } for p in trip]
                break

        return trip

    def __fix_mixed_modes(trip):
        weights = None
        for leg in trip:
            if leg["mode"] == "car+pt":
                if weights == None:
                    weights = [len([l["mode"] for l in trip if "car" in l["mode"]]),len([l["mode"] for l in trip if "pt" in l["mode"]  ])]
                leg["mode"] = random.choices(["car","pt"],weights=weights,k=1)[0]
        return trip

    @staticmethod
    def fix_trip(person):
        trip = person["legs"]
        attributes = person["attributes"]
        trip = sorted(trip, key=lambda x:x["arrival"])
        trip = TripCleaner.__fix_followed_homes(trip)
        trip = TripCleaner.__fix_trip_single_element(attributes, trip)
        trip = TripCleaner.__fix_missing_home(trip)
        trip = TripCleaner.__fix_weird_distances(trip)
        trip = TripCleaner.__fix_mixed_modes(trip)

        return trip