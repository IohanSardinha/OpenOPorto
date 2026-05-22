import time
import math
import random
import datetime
import pointpats
import cloudpickle
import numpy as np
import multiprocessing as mp
from shapely.geometry import Point
from ...ProcessStep import ProcessStep
from .travelSurvey import TravelSurveyGenericFormat
from concurrent.futures import ProcessPoolExecutor, as_completed

class HeuristicLocationAssigner(ProcessStep):
    
    def __init__(self, placesInGenericFormat, sections, placeCategoryMapper, home_id="home", silent=True, print_with_display=False):
        if print_with_display:
            from IPython.display import clear_output, display
            self.__clear_output = clear_output
            self.__display = display
        self.__silent=silent
        self.__print_with_display = print_with_display
        self.getPlaceCategory = placeCategoryMapper
        self.places = placesInGenericFormat.getPlaces()
        self.coords = placesInGenericFormat.getCoords()
        self.sections = sections
        self.home_id = home_id
        self.results = {}

    def print(self, *args, **kwargs):
        if not self.__silent:
            if self.__print_with_display:
                self.__display(*args, **kwargs)
            else:
                print(*args, **kwargs)
    
    def clear(self):
        if not self.__silent:
            if self.__print_with_display:
                self.__clear_output(True)
            else:
                print("\033c", end="")

    def sample_in_annulus(self, p0, target, alpha, polygon, attempts=100):
        for _ in range(attempts):
            θ = random.random() * 2*math.pi
            r = random.uniform(target-alpha, target+alpha)
            x = p0.x + r*math.cos(θ)
            y = p0.y + r*math.sin(θ)
            cand = Point(x,y)
            if polygon.contains(cand):
                return cand
        return None

    def build_candidates(self, person, trip):
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
                cand_ids.append(ids)
                is_discrete.append(True)
            else:
                cand_ids.append([None])
                is_discrete.append(False)
        return start, cand_ids, is_discrete

    def get_pt(self, is_discrete, sol_ids, sol_pts, i):
                if is_discrete[i]:
                    return self.coords[sol_ids[i]]
                return sol_pts[i]

    def validate(self, n , is_discrete, targets, sol_ids, sol_pts):
        err = 0
        for i in range(n):
            p1 = self.get_pt(is_discrete, sol_ids, sol_pts, i)
            p2 = self.get_pt(is_discrete, sol_ids, sol_pts, (i+1)%n)
            err += abs(p1.distance(p2) - targets[i])
        return err/n

    def hybrid_assign_iteration(self, n, start, is_discrete, cand_ids, targets, alpha, polygon, max_iters, startMoment, max_time_in_seconds):
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

    def hybrid_assign(self, person, trip, polygon, alpha=1500, max_iters=1000, restarts=100, max_time_in_seconds=1):
        startMoment = time.time()

        #THIS SHOULD NOT BE LIKE THIS
        sectionPoly = self.sections[self.sections["section"] == str(person.section)].iloc[0]["geometry"]
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
    
    def _process_person_wrapper(pickled_method, person, trips_legs, boundingBox, attempts, max_time_in_seconds, alpha):
        """Worker that unpickles and calls the method"""
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
        
        return (person[0], profilePlaces, err, fail)


    def process(self, persons, trips, boundingBox, attempts=100, max_time_in_seconds=0.3, num_workers=None):
        if not TravelSurveyGenericFormat().validate(trips):
            raise ValueError("Wrong format for travel survey")
        
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        # Serialize the method once
        pickled_method = cloudpickle.dumps(self.hybrid_assign)
        
        # Prepare work items
        work_items = []
        for person in persons.itertuples():
            person_id = person[-1]
            work_items.append((
                pickled_method,
                person,
                trips[person_id]["legs"],
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
            self.print("0%")
            startMoment = time.perf_counter()
        
        completed = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {executor.submit(_process_person_wrapper, *item): i 
                            for i, item in enumerate(work_items)}
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    person_id, profilePlaces, err, fail = future.result()
                    
                    if fail == 1:
                        count += 1
                        failed.append(str(idx))
                    elif fail == 2:
                        count += 1
                        exceptions += 1
                        failed.append(f"F->{idx}")
                    else:
                        results[person_id] = profilePlaces
                        if err is not None:
                            errors.append(err)
                    
                    completed += 1
                    
                    if not self.__silent:
                        elapsed = time.perf_counter() - startMoment
                        time_per_iter = elapsed / completed
                        remaining_seconds = (len(work_items) - completed) * time_per_iter
                        remaining = datetime.timedelta(seconds=int(remaining_seconds))
                        self.clear()
                        self.print(f"Heuristic Location Assigner\nProcessing{'.'*((completed//10%3)+1)}\n"
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
            self.print(failed)
        
        return self.results, errors