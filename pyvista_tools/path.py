import numpy as np
import pyvista as pv

def _convert_to_xyz(lon:np.ndarray, lat:np.ndarray, r:np.ndarray):
    """Converts longitude,latitudes and r (scale factor relative to earth's radius) into xyz coordinates"""
    x = r * np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    y = r * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(lon))
    z = r * np.sin(np.deg2rad(lat))
    return np.column_stack((x,y,z))

def _line_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points) - 1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points) - 1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

def camera_path(lon:np.ndarray, lat:np.ndarray, r:np.ndarray) -> pv.core.pointset.PolyData:
    """Creates a camera path/position defined by lon/lat position and the distance `r` (given in multiples of the earths radius)"""
    return _line_from_points(_convert_to_xyz(lon, lat, r))
    