import pickle
from collections.abc import Callable
from functools import wraps
from hashlib import sha1
from pathlib import Path
from typing import TypeVar, ParamSpec

from pyreproj import Reprojector
from shapely.geometry import Polygon

"""Utility helpers for caching and geometric bounding box creation.

This module provides a lightweight file-based cache decorator and a helper
class to build projected rectangular polygons from two coordinate pairs.
"""

P = ParamSpec("P")
R = TypeVar("R")

def cache(func: Callable[P, R]) -> Callable[P, R]:
    """Cache function results to disk based on argument hashing.

    The cache is stored under the ``cache/`` directory and can be globally
    disabled by setting ``__cache_enabled__ = False``.

    :param func: Target function to wrap.
    :type func: Callable[P, R]
    :returns: Wrapped function with persistent caching.
    :rtype: Callable[P, R]
    """

    name = func.__name__

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Execute cached call when available, otherwise compute and store.

        :param args: Positional arguments forwarded to the wrapped function.
        :type args: P.args
        :param kwargs: Keyword arguments forwarded to the wrapped function.
        :type kwargs: P.kwargs
        :returns: Wrapped function return value.
        :rtype: R
        """

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
    """Build rectangular bounding boxes in a target spatial reference system.

    :param origin_srs: Source spatial reference system identifier.
    :type origin_srs: str
    :param target_srs: Target spatial reference system identifier.
    :type target_srs: str
    """

    def __init__(self, origin_srs: str = "WGS84", target_srs: str = "EPSG:3763") -> None:
        """Initialize coordinate transformation settings.

        :param origin_srs: Source spatial reference system identifier.
        :type origin_srs: str
        :param target_srs: Target spatial reference system identifier.
        :type target_srs: str
        :returns: ``None``.
        :rtype: None
        """
        self.coordinateTransformer = Reprojector().get_transformation_function(from_srs=origin_srs, to_srs=target_srs)
        self.origin_srs = origin_srs
        self.target_srs = target_srs

    def build(self, long1: float, lat1: float, long2: float, lat2: float) -> Polygon:
        """Create a rectangular polygon from two opposite corner coordinates.

        If ``origin_srs`` and ``target_srs`` differ, the provided coordinates are
        reprojected before polygon construction.

        :param long1: Longitude (or x) of the first corner.
        :type long1: float
        :param lat1: Latitude (or y) of the first corner.
        :type lat1: float
        :param long2: Longitude (or x) of the opposite corner.
        :type long2: float
        :param lat2: Latitude (or y) of the opposite corner.
        :type lat2: float
        :returns: Axis-aligned rectangular polygon in target coordinates.
        :rtype: shapely.geometry.Polygon
        """
        if self.origin_srs != self.target_srs:
            x1, y1 = self.coordinateTransformer(lat1,long1)
            x2, y2 = self.coordinateTransformer(lat2,long2)
        else:
            x1, y1 = long1, lat1
            x2, y2 = long2, lat2
        return Polygon([[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]])