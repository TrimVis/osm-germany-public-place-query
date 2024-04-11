import osmnx as ox
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.transform import from_bounds
from rasterio.windows import bounds, transform as wtransform
from geopandas import GeoDataFrame
from dataclasses import dataclass
import shapely
import shapely.wkt
from typing import Any
from tqdm import tqdm
import gdal2tiles
import multiprocessing
import concurrent.futures as con

from config import debug
from public_places import extract_public_places


@dataclass(frozen=True)
class SmokeMask:
    forbidden: Any
    probably: Any


def create_smoke_mask(*, no_smoke_wkt, window, transform, window_transform):
    no_smoke_gdf = shapely.wkt.loads(no_smoke_wkt)
    theight, twidth = (window.height, window.width)

    if shapely.box(*bounds(window, transform)) \
            .intersects(no_smoke_gdf.geometry).any():

        no_smoke_mask = geometry_mask(no_smoke_gdf.geometry,
                                      transform=window_transform,
                                      out_shape=(theight, twidth),
                                      invert=True,
                                      all_touched=True)
        probably_smoke_mask = np.zeros((theight, twidth), dtype=np.bool_)

        return SmokeMask(no_smoke_mask, probably_smoke_mask)

    return None


def smoke_mask_data(public_places):
    # TODO pjordan: We should also take into consideration the buildings around it
    # Step 1: Find ground level of this building
    # (might need to be extracted in extract_public_places)
    # Step 2: Find all buildings in a 100m area around it's outline
    # (currently we only have knowledge of its center)
    # Step 3: Mark areas:
    #    a) Mark all areas as "potentially visible"  100m around the outline
    #    b) Mark all areas that have a clear direct viewline as "visible"

    # Simplified approach for now
    # Simply mark everything in a 100m area as no smoke
    shapes = [p.shape for p in public_places]
    no_smoke_gdf: Any = GeoDataFrame(geometry=shapes, crs="EPSG:4326")
    no_smoke_gdf = no_smoke_gdf.to_crs(epsg=32632)
    no_smoke_gdf.geometry = no_smoke_gdf.buffer(100)
    no_smoke_gdf = no_smoke_gdf.to_crs(epsg=4326)

    return no_smoke_gdf.to_wkt()


def create_germany_mask(*, germany_wkt, window, transform, window_transform):
    # TODO pjordan: Something is wrong with green in the top left corner
    germany_shape = shapely.wkt.loads(germany_wkt)
    theight, twidth = (window.height, window.width)

    if shapely.box(*bounds(window, transform)).intersects(germany_shape):
        return geometry_mask([germany_shape], invert=True,
                             transform=window_transform,
                             out_shape=(theight, twidth),
                             all_touched=True)
    return None


def germany_mask_data():
    # Download the boundary of Germany
    # germany_gdf = ox.geocode_to_gdf('R51477', by_osmid=True)
    germany_gdf = ox.geocode_to_gdf('Germany')

    # Get shape
    germany_shape = germany_gdf.unary_union
    return germany_shape.wkt


def compute_german_window(output_path, write_lock,
                          window, transform,
                          no_smoke_wkt, germany_wkt):
    # Compute all values
    world = np.full((4, window.height, window.width),
                    190, dtype=np.uint8)

    window_transform = wtransform(window, transform)
    smoke_mask = create_smoke_mask(
        no_smoke_wkt=no_smoke_wkt, window=window,
        transform=transform, window_transform=window_transform)
    germany_mask = create_germany_mask(
        germany_wkt=germany_wkt, window=window,
        transform=transform, window_transform=window_transform)

    if germany_mask is not None:
        # Mark germany as green initially
        world[0, germany_mask] = 0
        world[1, germany_mask] = 255
        world[2, germany_mask] = 0

    if smoke_mask:
        # Mark probably smoke zones
        world[0, smoke_mask.probably] = 255
        # Mark no smoke zones
        world[0, smoke_mask.forbidden] = 255
        world[1, smoke_mask.forbidden] = 0

    # Write out computed values at correct position
    with write_lock:
        if debug:
            print(f"Writing {window} to file")
        with rasterio.open(output_path, 'r+') as dst:
            dst.write(world, window=window)


def create_german_raster(*,
                         resolution, max_workers,
                         no_smoke_wkt, germany_wkt,
                         out_path):
    no_smoke_wkt = smoke_mask_data(public_places)
    germany_wkt = germany_mask_data()

    # Extract german bounds
    minx, miny, maxx, maxy = shapely.wkt.loads(germany_wkt).bounds
    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Raster metadata
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 4,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'LZW',
        'blockxsize': 3600,
        'blockysize': 3600,
        'tiled': True,
        'photometric': 'RGB',
    }

    print(" |> Creating germany raster")
    # Create a new file with wanted metadata
    with rasterio.open(out_path, 'w', **metadata) as dst:
        # Extract required windows
        windows = [window for _, window in dst.block_windows()]

    # write the actual tif file content across multiple processes
    with tqdm(total=len(windows)) as pbar:
        with multiprocessing.Manager() as man:
            write_lock = man.Lock()

            with con.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        compute_german_window,
                        out_path, write_lock,
                        window, transform,
                        no_smoke_wkt, germany_wkt,
                    ) for window in windows
                }
                for _ in con.as_completed(futures):
                    pbar.update(1)


def create_world_raster(*, width, height, out_path, germany_wkt):
    # World coverage with a simple pixel degree resolution
    degree = 360 / width
    transform = from_origin(-180, 90, degree, degree)
    german_box = shapely.box(*shapely.wkt.loads(germany_wkt).bounds)

    # Raster metadata
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 4,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'deflate',
        'blockxsize': 3600,
        'blockysize': 3600,
        'tiled': True,
        'photometric': 'RGB',
    }

    print(" |> Creating world raster")
    # Create a new tif file with wanted metadata
    with rasterio.open(out_path, 'w', **metadata) as dst:
        windows = [window for _, window in dst.block_windows()]
        for window in tqdm(windows):
            window_transform = wtransform(window, transform)

            # if it is outside of our german box, make it grey
            dim = (4, window.height, window.width)
            world = np.full(dim, 0, dtype=np.uint8)

            # Color all non-german area grey
            mask = geometry_mask([german_box], transform=window_transform,
                                 out_shape=(window.height, window.width),
                                 all_touched=True)
            world[:, mask] = 190
            world[:, ~mask] = 1

            dst.write(world, window=window)


def create_tiles(tif_path, out_dir, *, zoom="0-3", max_workers=8, no_data=None):
    gdal2tiles.generate_tiles(tif_path, out_dir,
                              resume=False, profile="mercator",
                              resampling='average', kml=False,
                              srcnodata=no_data, zoom=zoom,
                              nb_processes=max_workers,)


# Example usage
if __name__ == "__main__":
    _create_tifs = True
    _create_tiles = True
    _create_world = False
    _create_germany = True

    if _create_tifs:
        location = "Waldshut, Baden-Wuerttemberg, Germany"
        public_places = extract_public_places(location)

        print(" |> Extracting mask data")
        no_smoke_wkt = smoke_mask_data(public_places)
        germany_wkt = germany_mask_data()

        print(" |> Creating rasters...")
        if _create_world:
            create_world_raster(width=14400,
                                height=7200,
                                germany_wkt=germany_wkt,
                                out_path="world_map.tif")
        if _create_germany:
            create_german_raster(resolution=0.0001,
                                 max_workers=8,
                                 no_smoke_wkt=no_smoke_wkt,
                                 germany_wkt=germany_wkt,
                                 out_path="germany_map.tif")

    if _create_tiles:
        print(" |> Creating tiles...")
        if _create_world:
            print(" |> Creating world tiles...")
            create_tiles("world_map.tif", "world_map/", zoom="0-5", no_data="1")
        if _create_germany:
            print(" |> Creating germany tiles...")
            create_tiles("germany_map.tif", "germany_map/", zoom="0-4")
