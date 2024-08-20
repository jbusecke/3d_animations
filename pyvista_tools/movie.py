import pyvista as pv
import geovista as gv
from dataclasses import dataclass
from tqdm.auto import tqdm
import xarray as xr

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
    camera_path:pv.core.pointset.PolyData = None

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
        
        # infer frames
        self.frames = range(len(self.da[self.time_dim]))        
        

    def _get_plotter(self,**geoplotter_kwargs):
        p = gv.GeoPlotter(**geoplotter_kwargs)
        p.add_base_layer(color=self.base_color)
        p.background_color=self.background_color
        return p

    def _get_empty_mesh(self):
        return gv.Transform.from_2d(
            self.da[self.lon_name], 
            self.da[self.lat_name]
        )

    def _get_frame(self,frame):
        da_frame = self.da.isel({self.time_dim:frame}).drop_vars('time')
        nanmask = ~np.isnan(da_frame).reset_coords(drop=True)

        for co in [self.lon_name, self.lat_name]:
            da_frame = da_frame.assign_coords({co:da_frame.coords[co].where(nanmask)})
        return da_frame

    def _update_mesh(self,mesh, da_frame, plotter, name='data-frame'):
        # set the active scalar with the data payload for the frame
        mesh[name] = da_frame.data.flatten()

        # threshold the mesh - this can change the mesh geometry each frame,
        # so create a separate "frame" mesh
        frame_mesh = mesh.threshold()

        # this will add the named "data-frame" mesh to the plotter or replace it
        actor = plotter.add_mesh(
            frame_mesh, 
            name=name,
            cmap=self.cmap,
            clim=self.clim,
        )
        return actor

    def set_frame(self, frame, plotter, mesh):
        da_frame = self._get_frame(frame)
        a = self._update_mesh(mesh, da_frame, plotter)
        if self.camera_path:
            self._update_camera(frame, plotter)
        return a
        
        

    def _update_camera(self, frame, plotter):
        plotter.set_position(self.camera_path.points[frame, :])
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.roll = -90
        
    def preview(self, frame):
        pv.global_theme.trame.interactive_ratio = 0.75 #1 or 2 
        pv.global_theme.trame.still_ratio = 4
        window_size = None
        p = self._get_plotter(window_size=window_size)
        mesh = self._get_empty_mesh()
        a = self.set_frame(frame, p, mesh)
        
        p.show()

    def render(self, filename, resolution: list=[3840, 2160]):
        #(does this affect the movie quality?)
        pv.global_theme.trame.interactive_ratio = 5 #1 or 2 (does this affect the movie quality?)
        pv.global_theme.trame.still_ratio = 5
        p = self._get_plotter(window_size=(resolution))
        mesh = self._get_empty_mesh()
        p.open_movie(filename)
        for frame in tqdm(self.frames):
            a = self.set_frame(frame, p, mesh)
            da_frame = self._get_frame(frame)
            p.write_frame()
        p.close()