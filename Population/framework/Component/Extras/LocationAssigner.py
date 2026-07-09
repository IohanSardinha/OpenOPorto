"""Heuristic location assignment utilities."""

from typing import Any

import time
import math
import random
import datetime
import pointpats
import cloudpickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from pyreproj import Reprojector
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor, as_completed
from ...misc import cache

class PlacesGenericFormat():
    """Load a places file and expose transformed coordinates.
    
    Methods
    -------
    getPlaces() -> pd.DataFrame
        Return the loaded places table.
    getCoords() -> dict[int, Point]
        Return the places with transformed coordinates.
    """

    def __init__(self, placesFile: str, origin_srs: str = "WGS84", target_srs: str = "EPSG:3763") -> PlacesGenericFormat:
        """Read places and precompute projected coordinates.
        
        :param placesFile: Path to a CSV file with columns 'latitude', 'longitude', and 'category'.
        :type placesFile: str
        :param origin_srs: Source spatial reference system identifier.
        :type origin_srs: str
        :param target_srs: Target spatial reference system identifier.
        :type target_srs: str
        :returns: An instance of PlacesGenericFormat with loaded places and transformed coordinates.
        :rtype: PlacesGenericFormat
        """
        self.coordinateTransformer = Reprojector().get_transformation_function(from_srs=origin_srs, to_srs=target_srs)
        self.origin_srs = origin_srs
        self.target_srs = target_srs
        self.__places = pd.read_csv(placesFile)
        self.__places["x"] = self.__places.apply(lambda x: self.coordinateTransformer(x["latitude"],x["longitude"])[0],axis=1)
        self.__places["y"] = self.__places.apply(lambda x: self.coordinateTransformer(x["latitude"],x["longitude"])[1],axis=1)
        self.__places = self.__places.reset_index().rename(columns={'index':'id'})

        self.__coords = {}
        for _, row in self.__places.iterrows():
            self.__coords[row['id']] = Point(row['x'], row['y'])

    def getPlaces(self) -> pd.DataFrame:
        """Return the loaded places table.
        
        :returns: DataFrame of places with transformed coordinates.
        :rtype: pd.DataFrame
        """
        return self.__places

    def getCoords(self) -> dict[int, Point]:
        """Return the places with transformed coordinates.
        
        :returns: Dictionary mapping place IDs to their transformed coordinates.
        :rtype: dict[int, Point]
        """
        return self.__coords

class HeuristicLocationAssigner():
    """Assign locations to activity chains using heuristic search.
    
    Methods
    -------
    hybrid_assign(person: Any, trip: list[dict[str, Any]], polygon: Any, alpha: float = 1500, max_iters: int = 1000, restarts: int = 100, max_time_in_seconds: float = 1) -> tuple[list[Any], float]
        Assign locations to one person's trip chain.
    process(population: pd.DataFrame, boundingBox: Any, attempts: int = 100, max_time_in_seconds: float = 0.3, num_workers: int | None = None) -> pd.DataFrame
        Assign locations to every person in a population table.
    """
    
    def __init__(self, placesFile: str, sections: pd.DataFrame, placeCategoryMapper: Any, home_id: str = "home", silent: bool = True, print_with_display: bool = False) -> HeuristicLocationAssigner:
        """Create a heuristic location assigner.
        
        :param placesFile: Path to the file containing place data.
        :type placesFile: str
        :param sections: DataFrame containing activity sections.
        :type sections: pd.DataFrame
        :param placeCategoryMapper: Function to map activities to place categories.
        :type placeCategoryMapper: Any
        :param home_id: ID of the home location.
        :type home_id: str
        :param silent: Whether to run in silent mode.
        :type silent: bool
        :param print_with_display: Whether to use IPython display for output.
        :type print_with_display: bool
        :returns: An instance of HeuristicLocationAssigner.
        :rtype: HeuristicLocationAssigner
        """
        if print_with_display:
            from IPython.display import clear_output, display
            self.__clear_output = clear_output
            self.__display = display
        self.__silent=silent
        self.__print_with_display = print_with_display
        self.getPlaceCategory = placeCategoryMapper
        placesInGenericFormat = PlacesGenericFormat(placesFile)
        self.places = placesInGenericFormat.getPlaces()
        self.coords = placesInGenericFormat.getCoords()
        self.sections = sections
        self.home_id = home_id
        self.results = {}

    def __print(self, *args: Any, **kwargs: Any) -> None:
        """Print progress output unless running in silent mode.
        
        :param args: Positional arguments to print.
        :type args: Any
        :param kwargs: Keyword arguments to print.
        :type kwargs: Any
        """
        if not self.__silent:
            if self.__print_with_display:
                self.__display(*args, **kwargs)
            else:
                print(*args, **kwargs)
    
    def __clear(self) -> None:
        """Clear the current progress display unless silent."""
        if not self.__silent:
            if self.__print_with_display:
                self.__clear_output(True)
            else:
                print("\033c", end="")

    def sample_in_annulus(self, p0: Point, target: float, alpha: float, polygon: Any, attempts: int = 100) -> Point | None:
        """Sample a point near a target distance inside a polygon.
        
        :param p0: Reference point.
        :type p0: Point
        :param target: Target distance from the reference point.
        :type target: float
        :param alpha: Allowable deviation from the target distance.
        :type alpha: float
        :param polygon: Polygon within which to sample.
        :type polygon: Any
        :param attempts: Number of sampling attempts.
        :type attempts: int
        :returns: A sampled point or None if no valid point found.
        :rtype: Point | None
        """
        for _ in range(attempts):
            θ = random.random() * 2*math.pi
            r = random.uniform(target-alpha, target+alpha)
            x = p0.x + r*math.cos(θ)
            y = p0.y + r*math.sin(θ)
            cand = Point(x,y)
            if polygon.contains(cand):
                return cand
        return None

    def build_candidates(self, person: Any, trip: list[dict[str, Any]]) -> tuple[int | None, list[list[Any]], list[bool]]:
        """Build candidate place identifiers for each trip leg.
        
        :param person: The person for whom to build candidates.
        :type person: Any
        :param trip: The trip for which to build candidates.
        :type trip: list[dict[str, Any]]
        :returns: A tuple containing the start index, candidate IDs, and discrete flags.
        :rtype: tuple[int | None, list[list[Any]], list[bool]]
        """
        cand_ids = []
        is_discrete = []
        start = None
        for idx, leg in enumerate(trip):
            act = leg["activity"]
            cat = self.getPlaceCategory(act, person)
            if act==self.home_id:
                start = idx if start == None else start
                cand_ids.append([self.home_id])
                is_discrete.append(True)
            elif cat!="ALL":
                ids = self.places.loc[self.places["category"].isin(cat), 'id'].tolist()
                if len(ids) == 0:
                    cand_ids.append([None])
                    is_discrete.append(False)
                else:
                    cand_ids.append(ids)
                    is_discrete.append(True)
            else:
                cand_ids.append([None])
                is_discrete.append(False)
        return start, cand_ids, is_discrete

    def get_pt(self, is_discrete: list[bool], sol_ids: list[Any], sol_pts: dict[int, Any], i: int) -> Any:
        """Return the point associated with a leg solution index.
        
        :param is_discrete: List indicating if each leg is discrete.
        :type is_discrete: list[bool]
        :param sol_ids: List of selected place IDs for discrete legs.
        :type sol_ids: list[Any]
        :param sol_pts: Dictionary of sampled points for continuous legs.
        :type sol_pts: dict[int, Any]
        :param i: Index of the leg to retrieve.
        :type i: int
        :returns: The point corresponding to the leg index.
        :rtype: Any
        """
        if is_discrete[i]:
            return self.coords[sol_ids[i]]
        return sol_pts[i]

    def validate(self, n: int, is_discrete: list[bool], targets: list[float], sol_ids: list[Any], sol_pts: dict[int, Any]) -> float:
        """Score a solution by the average distance error.
        
        :param n: Number of legs in the trip.
        :type n: int
        :param is_discrete: List indicating if each leg is discrete.
        :type is_discrete: list[bool]
        :param targets: List of target distances for each leg.
        :type targets: list[float]
        :param sol_ids: List of selected place IDs for discrete legs.
        :type sol_ids: list[Any]
        :param sol_pts: Dictionary of sampled points for continuous legs.
        :type sol_pts: dict[int, Any]
        :returns: Average distance error across all legs.
        :rtype: float
        """
        err = 0
        for i in range(n):
            p1 = self.get_pt(is_discrete, sol_ids, sol_pts, i)
            p2 = self.get_pt(is_discrete, sol_ids, sol_pts, (i+1)%n)
            err += abs(p1.distance(p2) - targets[i])
        return err/n

    def hybrid_assign_iteration(self, n: int, start: int | None, is_discrete: list[bool], cand_ids: list[list[Any]], targets: list[float], alpha: float, polygon: Any, max_iters: int, startMoment: float, max_time_in_seconds: float) -> tuple[tuple[list[Any], dict[int, Any]], float]:
        """Run one search iteration for a single person's trip chain.
        
        :param n: Number of legs in the trip.
        :type n: int
        :param start: Starting index for the search.
        :type start: int | None
        :param is_discrete: List indicating if each leg is discrete.
        :type is_discrete: list[bool]
        :param cand_ids: List of candidate place IDs for discrete legs.
        :type cand_ids: list[list[Any]]
        :param targets: List of target distances for each leg.
        :type targets: list[float]
        :param alpha: Allowable deviation from target distances.
        :type alpha: float
        :param polygon: Polygon within which to sample continuous points.
        :type polygon: Any
        :param max_iters: Maximum number of iterations for the search.
        :type max_iters: int
        :param startMoment: Timestamp when the search started.
        :type startMoment: float
        :param max_time_in_seconds: Maximum allowed time for the search.
        :type max_time_in_seconds: float
        :returns: A tuple containing the best solution and its error.
        :rtype: tuple[tuple[list[Any], dict[int, Any]], float]
        """
        i = start
        sol_ids = [None]*n
        sol_pts = {}

        for _ in range(n):
            prevIdx = (i-1)%n
            
            if is_discrete[i]:
                sol_ids[i] = random.choice(cand_ids[i])
            else:
                prev_pt = sol_pts.get(prevIdx) or self.coords[sol_ids[prevIdx]]
                art = self.sample_in_annulus(prev_pt, targets[i], alpha, polygon)
                sol_pts[i] = art
            i = (i+1)%n

        err = self.validate(n, is_discrete, targets, sol_ids, sol_pts)
        best_sol = (sol_ids.copy(), sol_pts.copy())

        for _ in range(max_iters):
            if time.time() - startMoment > max_time_in_seconds: break
            i = random.randrange(n)
            if is_discrete[i]:
                # try swapping to another place
                new_id = random.choice(cand_ids[i])
                old_id = sol_ids[i]
                if new_id == old_id: continue
                sol_ids[i] = new_id
            else:
                # resample a new random point
                prev = self.get_pt(is_discrete, sol_ids, sol_pts, (i-1)%n)
                new_pt = self.sample_in_annulus(prev, targets[i], alpha, polygon)
                if new_pt is None: continue
                old_pt = sol_pts.get(i)
                sol_pts[i] = new_pt

            new_err = self.validate(n, is_discrete, targets, sol_ids, sol_pts)
            if new_err < err:
                err = new_err
                best_sol = (sol_ids.copy(), sol_pts.copy())
            else:
                # revert
                if is_discrete[i]:
                    sol_ids[i] = old_id
                else:
                    sol_pts[i] = old_pt

            if err <= alpha:
                break
        return best_sol, err
    
    def hybrid_assign(self, person: Any, trip: list[dict[str, Any]], polygon: Any, alpha: float = 1500, max_iters: int = 1000, restarts: int = 100, max_time_in_seconds: float = 1) -> tuple[list[Any], float]:
        """Assign locations to one person's trip chain.
        
        :param person: The person for whom to assign locations.
        :type person: Any
        :param trip: The trip for which to assign locations.
        :type trip: list[dict[str, Any]]
        :param polygon: Polygon within which to sample continuous points.
        :type polygon: Any
        :param alpha: Allowable deviation from target distances.
        :type alpha: float
        :param max_iters: Maximum number of iterations for the search.
        :type max_iters: int
        :param restarts: Number of random restarts for the search.
        :type restarts: int
        :param max_time_in_seconds: Maximum allowed time for the search.
        :type max_time_in_seconds: float
        :returns: A tuple containing the assigned locations and the error.
        :rtype: tuple[list[Any], float]
        """
        startMoment = time.time()

        #THIS SHOULD NOT BE LIKE THIS
        sectionPoly = self.sections[self.sections["section"] == str(person["section"])].iloc[0]["geometry"]
        home = Point(pointpats.random.poisson(sectionPoly,size=1))

        self.coords[self.home_id] = home

        n = len(trip)
        start, cand_ids, is_discrete = self.build_candidates(person, trip)

        targets = [float(tp["distance"]) for tp in trip]
        best_sol, best_err = None, float('inf')

        for _ in range(restarts):
            if time.time() - startMoment > max_time_in_seconds: 
                break

            sol, err = self.hybrid_assign_iteration(n, start, is_discrete, cand_ids, targets, alpha, polygon, max_iters, startMoment, max_time_in_seconds)

            if err < best_err:
                best_err = err
                best_sol = sol

            if best_err <= alpha:
                break

        if best_sol is None:
            return [], float('inf')
        
        out = []
        for i in range(n):
            if is_discrete[i]:
                out.append(self.coords[best_sol[0][i]])
            else:
                out.append(best_sol[1][i])

        return out,best_err
    
    def process(self, population: pd.DataFrame, boundingBox: Any, attempts: int = 100, max_time_in_seconds: float = 0.3, num_workers: int | None = None) -> pd.DataFrame:
        """Assign locations to every person in a population table.
        
        :param population: DataFrame of persons with trip chains.
        :type population: pd.DataFrame
        :param boundingBox: Polygon within which to sample continuous points.
        :type boundingBox: Any
        :param attempts: Number of attempts per person.
        :type attempts: int
        :param max_time_in_seconds: Maximum allowed time per person.
        :type max_time_in_seconds: float
        :param num_workers: Number of parallel workers to use.
        :type num_workers: int | None
        :returns: DataFrame of persons with assigned locations.
        :rtype: pd.DataFrame
        """
        
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        # Serialize the method once
        pickled_method = cloudpickle.dumps(self.hybrid_assign)
        
        # Prepare work items
        work_items = []
        for i, person in enumerate(population.to_dict("records")):

            leg_keys = set(["_".join(key.split("_")[2:]) for key in person.keys() if key.startswith("leg_") and not key == "leg_count"])
            legs = [{key: person[f"leg_{i}_{key}"] for key in leg_keys} for i in range(person["leg_count"]) if person[f"leg_{i}_activity"] is not None]

            work_items.append((
                pickled_method,
                person,
                legs,
                boundingBox,
                attempts,
                max_time_in_seconds,
                1000  # alpha
            ))
        
        count = 0
        exceptions = 0
        failed = []
        errors = []
        results = {}
        
        if not self.__silent:
            self.__print("0%")
            startMoment = time.perf_counter()
        
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(_process_person_wrapper, *item): i 
                            for i, item in enumerate(work_items)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    profilePlaces, err, fail = future.result()
                    
                    if fail == 1:
                        count += 1
                        failed.append(str(idx))
                    elif fail == 2:
                        count += 1
                        exceptions += 1
                        failed.append(f"F->{idx}")
                    else:
                        results[idx] = profilePlaces
                        if err is not None:
                            errors.append(err)
                    
                    completed += 1
                    
                    if not self.__silent:
                        elapsed = time.perf_counter() - startMoment
                        time_per_iter = elapsed / completed
                        remaining_seconds = (len(work_items) - completed) * time_per_iter
                        remaining = datetime.timedelta(seconds=int(remaining_seconds))
                        self.__clear()
                        self.__print(f"Heuristic Location Assigner\nProcessing{'.'*((completed//10%3)+1)}\n"
                                f"Completed: {round(100*completed/len(work_items),4)}%, "
                                f"Failed: {100*count/completed if completed > 0 else 0}%, "
                                f"Exceptions: {(100*exceptions/count) if count > 0 else 0}%\n"
                                f"Expected remaining Time: {remaining}")
                        
                except KeyboardInterrupt:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as e:
                    print(f"Error processing person {idx}: {e}")
                    count += 1
                    exceptions += 1
                    failed.append(f"F->{idx}")
        
        errors = np.array(errors) if errors else np.array([])
        self.results = results
        
        if len(failed) > 0:
            self.__print(failed)
        
        return self.match_results(population, results)
    
    def match_results(self, population: pd.DataFrame, locations: dict[int, list[Point]]) -> pd.DataFrame:
        """Write matched coordinates back to the population dataframe.
        
        :param population: Original population dataframe.
        :type population: pd.DataFrame
        :param locations: Dictionary mapping person indices to lists of assigned Points.
        :type locations: dict[int, list[Point]]
        :returns: Updated population dataframe with assigned coordinates.
        :rtype: pd.DataFrame
        """

        for i in range(population["leg_count"].max()):
            x = [points[i].x if i < len(points) else None for _, points in sorted([(j, p) for j, p in locations.items()], key=lambda x: x[0])]
            y = [points[i].y if i < len(points) else None for _, points in sorted([(j, p) for j, p in locations.items()], key=lambda x: x[0])]
            print("x,y:",len(x), len(y))
            population[f"leg_{i}_x"] = x
            population[f"leg_{i}_y"] = y
        
        return population

def _process_person_wrapper(pickled_method: bytes, person: Any, trips_legs: list[dict[str, Any]], boundingBox: Any, attempts: int, max_time_in_seconds: float, alpha: float) -> tuple[list[Any], Any, int]:
    """Worker that unpickles and calls the method.
    
    :param pickled_method: Serialized hybrid_assign method.
    :type pickled_method: bytes
    :param person: The person for whom to assign locations.
    :type person: Any
    :param trips_legs: List of trip legs for the person.
    :type trips_legs: list[dict[str, Any]]
    :param boundingBox: Polygon within which to sample continuous points.
    :type boundingBox: Any
    :param attempts: Number of attempts per person.
    :type attempts: int
    :param max_time_in_seconds: Maximum allowed time per person.
    :type max_time_in_seconds: float
    :param alpha: Allowable deviation from target distances.
    :type alpha: float
    :returns: A tuple containing the assigned locations, error, and failure code.
    :rtype: tuple[list[Any], Any, int]
    """
    hybrid_assign = cloudpickle.loads(pickled_method)
    
    fail = 0
    profilePlaces = []
    err = None
    
    for _ in range(attempts):
        try:
            profilePlaces, err = hybrid_assign(
                person, trips_legs, boundingBox, 
                alpha=alpha, max_time_in_seconds=max_time_in_seconds
            )
            if len(profilePlaces) == 0:
                fail = 1
            else:
                fail = 0
                break
        except KeyboardInterrupt:
            raise
        except Exception:
            fail = 2
    
    return (profilePlaces, err, fail)