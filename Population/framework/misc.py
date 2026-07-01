import pickle
from hashlib import sha1
from pathlib import Path
from pyreproj import Reprojector
from shapely.geometry import Polygon

def cache(func):

    name = func.__name__

    def wrapper(*args, **kwargs):

        if not globals().get('__cache_enabled__', True): return func(*args, **kwargs)

        file_str = f"{'|'.join(str(arg) for arg in args)}|{'|'.join(f'{key}|{value}' for key, value in kwargs.items())}"
        file_name = f"cache/{name}{sha1(pickle.dumps(file_str)).hexdigest()}.pkl"

        if Path(file_name).exists():
            with open(file_name, "rb") as f:
                print(f"Cache hit for {file_name}")
                return pickle.load(f)

        print(f"Cache miss for {file_name}")
        cache = func(*args, **kwargs)
        
        with open(file_name, 'wb') as f:
            f.write(pickle.dumps(cache))
            return cache
        
    return wrapper

class BoundingBoxBuilder():
    def __init__(self, origin_srs="WGS84", target_srs="EPSG:3763"):
        self.coordinateTransformer = Reprojector().get_transformation_function(from_srs=origin_srs, to_srs=target_srs)
        self.origin_srs = origin_srs
        self.target_srs = target_srs

    def build(self, long1, lat1, long2, lat2):
        if self.origin_srs != self.target_srs:
            x1, y1 = self.coordinateTransformer(lat1,long1)
            x2, y2 = self.coordinateTransformer(lat2,long2)
        else:
            x1, y1 = long1, lat1
            x2, y2 = long2, lat2
        return Polygon([[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]])