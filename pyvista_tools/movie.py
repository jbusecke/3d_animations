import pyvista as pv
import geovista as gv
from dataclasses import dataclass
from tqdm.auto import tqdm
import xarray as xr
import numpy as np

@dataclass
class Movie:
    da: xr.DataArray
    cmap:str = 'inferno'
    clim:list = None
    x_dim:str = 'x'
    y_dim:str = 'y'
    time_dim:str = 'time'
    lon_name:str = 'lon'
    lat_name:str = 'lat'
    background_color:str = 'black'
    base_color = 'gray'
    base_texture=None
    camera_path:pv.core.pointset.PolyData = None
    geoplotter_kwargs=None


    def __post_init__(self):
        dims = set([self.x_dim, self.y_dim])
        coords = set([self.lon_name, self.lat_name])
        if not dims.issubset(self.da.dims):
            raise ValueError(f"Could not find dimension(s){dims-set(self.da.dims)}. Got {list(self.da.dims)}")
        if not coords.issubset(self.da.coords):
            raise ValueError(f"Could not find coordinates{coords-set(self.da.coords)}. Got {list(self.da.coords)}")
        
        if any(len(self.da[co].dims)!=2 for co in coords):
            raise ValueError(f"Expected longitude and latitude coordinates to be 2 dimensional. Got {[(co, list(da[co].dims)) for co in coords]}")

            # It seems like some of the internal steps (`make_mesh` and in particular gv.Transform.from_2d(lon, lat, data=da.data) seem to assume that)
        # the dimensions are ordered in a certain way! 
        # TODO: Raise an issue and see if we can fix this upstream
        # for now always transpose here
        other_dims = [dim for dim in list(self.da.dims) if dim not in [self.time_dim, self.x_dim, self.y_dim]]
        self.da = self.da.transpose(*['x', 'y', 'time']+other_dims)
        # I am not subsampling yet, but I should probably init it here to be consitent?

        #Assume that the nan mask does not change
        self.nanmask = ~np.isnan(self.da.isel({self.time_dim:0})).reset_coords(drop=True).load()

        # mask out the coordinates
        for co in [self.lon_name, self.lat_name]:
            self.da = self.da.assign_coords({co:self.da.coords[co].where(self.nanmask)})
        
        # infer frames
        self.frames = range(len(self.da[self.time_dim]))

        # set defaults
        if self.geoplotter_kwargs is None:
            self.geoplotter_kwargs = {}

        self.plotter = gv.GeoPlotter(**self.geoplotter_kwargs)
        if self.base_color or self.base_texture:
            self.plotter.add_base_layer(color=self.base_color, texture=self.base_texture)            

    def _get_empty_mesh(self):
        mesh = gv.Transform.from_2d(
            self.da[self.lon_name], 
            self.da[self.lat_name]
        )
        print(mesh)
        print(self.nanmask.data.flatten().shape)
        # mesh.remove_cells(self.nanmask.data.flatten(), inplace=True)
        return mesh

    def _get_frame(self,frame):
        da_frame = self.da.isel({self.time_dim:frame}).drop_vars('time')
        return da_frame

    def _update_mesh(self,mesh, da_frame):
        # set the active scalar with the data payload for the frame
        # get name from xarray
        mesh[da_frame.name] = da_frame.data.flatten()

        # # threshold the mesh - this can change the mesh geometry each frame,
        # # so create a separate "frame" mesh
        # frame_mesh = mesh.threshold()
        frame_mesh = mesh

        # this will add the named "data-frame" mesh to the plotter or replace it
        actor = self.plotter.add_mesh(
            frame_mesh, 
            cmap=self.cmap,
            clim=self.clim,
        )
        return actor

    def set_frame(self, frame, mesh):
        da_frame = self._get_frame(frame)
        a = self._update_mesh(mesh, da_frame)
        if self.camera_path:
            self._update_camera(frame)
        return a

    def _update_camera(self, frame):
        self.plotter.set_position(self.camera_path.points[frame, :])
        self.plotter.camera.focal_point = (0, 0, 0)
        self.plotter.camera.roll = -90
        
    def preview(self, frame):
        pv.global_theme.trame.interactive_ratio = 0.75 #1 or 2 
        pv.global_theme.trame.still_ratio = 4

        mesh = self._get_empty_mesh()
        a = self.set_frame(frame, mesh)
        p.show()

    def render(self, filename, resolution: list=[1920, 1088]):
        #(does this affect the movie quality?)
        pv.global_theme.trame.interactive_ratio = 5 #1 or 2 (does this affect the movie quality?)
        pv.global_theme.trame.still_ratio = 5
        # set the window size to the given resolution
        self.plotter.window_size = resolution
        mesh = self._get_empty_mesh()
        self.plotter.open_movie(filename)
        for frame in tqdm(self.frames):
            a = self.set_frame(frame, mesh)
            da_frame = self._get_frame(frame)
            self.plotter.write_frame()
        self.plotter.close()