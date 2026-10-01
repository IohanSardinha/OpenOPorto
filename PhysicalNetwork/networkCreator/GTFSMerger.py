import gtfs_kit as gk
import pandas as pd
from pathlib import Path

class GTFSMerger():

    schedules = {}

    @staticmethod
    def __get_timezone():
        path = Path("/etc/localtime").resolve()

        prefix = Path("/usr/share/zoneinfo/")
        if path.is_relative_to(prefix):
            timezone = str(path.relative_to(prefix))
            return timezone
        return "UTC"

    def add_gtfs(self, file_name, name, units="km", start_date="20260101", end_date="20261231",service_filter=None, route_type=None, agency_id=None):

        print(file_name, name, units, start_date, end_date, service_filter, route_type, agency_id)

        self.units = units

        schedule = gk.read_feed(file_name, dist_units=units)

        weekday_filter =(schedule.calendar[["monday", "tuesday", "wednesday", "thursday", "friday"]] == [1, 1, 1, 1, 1]).all(axis=1)
        schedule.calendar = schedule.calendar[weekday_filter]
        if service_filter is not None: schedule.calendar = schedule.calendar[schedule.calendar["service_id"] == service_filter]
        schedule.trips = schedule.trips[schedule.trips["service_id"].isin(schedule.calendar["service_id"])]
        if agency_id is not None: schedule.routes["agency_id"] = agency_id
        schedule.stop_times.drop_duplicates(subset=["trip_id", "stop_id"], inplace=True)
        schedule.stop_times = schedule.stop_times.sort_values("stop_id", ascending=False).drop_duplicates(subset=["trip_id", "departure_time"])
        schedule.stop_times = schedule.stop_times[schedule.stop_times["trip_id"].isin(schedule.trips["trip_id"])]
        schedule.stops = schedule.stops[schedule.stops["stop_id"].isin(schedule.stop_times["stop_id"])]
        schedule.routes = schedule.routes[schedule.routes["route_id"].isin(schedule.trips["route_id"])]
        schedule.calendar["start_date"] = start_date
        schedule.calendar["end_date"] = end_date
        if schedule.calendar_dates is not None: schedule.calendar_dates = None #schedule.calendar_dates[schedule.calendar_dates.index == -1]
        if route_type is not None: schedule.routes["route_type"] = route_type
        schedule.stop_times.loc[schedule.stop_times["arrival_time"].str.match(r'^\d{2}:\d{2}$'), "arrival_time"] += ":00"
        schedule.stop_times.loc[schedule.stop_times["departure_time"].str.match(r'^\d{2}:\d{2}$'), "departure_time"] += ":00"
        schedule.trips["direction_id"] = schedule.trips["direction_id"]%2
        if schedule.agency is None: schedule.agency = pd.DataFrame([{"agency_id": name, "agency_name": name, "agency_url": "http://example.com", "agency_timezone": self.__get_timezone()}])
        schedule.agency["agency_id"] = name

        schedule.routes["route_long_name"] = schedule.routes["route_short_name"].where(schedule.routes["route_long_name"].isna(), schedule.routes["route_long_name"])
        n = schedule.routes.groupby(["route_short_name","route_long_name"]).cumcount()+1
        counts = schedule.routes.groupby(["route_short_name","route_long_name"])["route_long_name"].transform("size")
        schedule.routes["route_long_name"] = schedule.routes["route_long_name"].astype(str) + "_" + n.astype(str).where(counts>1, "")
        self.schedules[name] = schedule

    def merge_gtfs(self):
        for name, feed in self.schedules.items():

            feed.trips[["route_id", "service_id", "trip_id", "shape_id"]] = name + "_" + feed.trips[["route_id", "service_id", "trip_id", "shape_id"]]

            feed.routes["route_id"] = name + "_" + feed.routes["route_id"]

            feed.stops[["stop_id", "stop_code"]] = name + "_" + feed.stops[["stop_id", "stop_code"]]

            if feed.shapes is not None:
                feed.shapes["shape_id"] = name + "_" + feed.shapes["shape_id"]

            feed.stop_times[["trip_id","stop_id"]] = name + "_" + feed.stop_times[["trip_id","stop_id"]]

            feed.routes["agency_id"] = name
            feed.calendar["service_id"] = name + "_" + feed.calendar["service_id"]

        agencies = pd.concat([feed.agency for feed in self.schedules.values()], ignore_index=True)
        calendar = pd.concat([feed.calendar for feed in self.schedules.values()], ignore_index=True)
        routes = pd.concat([feed.routes for feed in self.schedules.values()], ignore_index=True)
        stops = pd.concat([feed.stops for feed in self.schedules.values()], ignore_index=True)
        shapes = pd.concat([feed.shapes for feed in self.schedules.values() if feed.shapes is not None], ignore_index=True)
        stop_times = pd.concat([feed.stop_times for feed in self.schedules.values()], ignore_index=True)
        trips = pd.concat([feed.trips for feed in self.schedules.values()], ignore_index=True)

        self.merged_gtfs = gk.feed.Feed(dist_units=self.units, agency=agencies, calendar=calendar, routes=routes, stops=stops, shapes=shapes, stop_times=stop_times, trips=trips)

        return self.merged_gtfs

    def save_merged_gtfs(self, output_path):
        self.merged_gtfs.write(output_path)