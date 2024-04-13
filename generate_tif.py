import gc

import osmnx as ox
import numpy as np
import rasterio
import shapely
import shapely.wkt
import multiprocessing
import concurrent.futures as con
import pandas as pd

from pathlib import Path
from rasterio.features import geometry_mask
from rasterio.transform import from_origin, from_bounds
from rasterio.windows import bounds, transform as wtransform
from geopandas import GeoDataFrame
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

from config import debug
from public_places import extract_public_places
from pedestrian_zones import extract_pedestrian_zones

try:
    # try to use the custom version that ignores error messages
    # due to a known gdal issue on arch linux
    import gdal2tiles_custom.gdal2tiles as gdal2tiles
except ImportError:
    import gdal2tiles


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


def smoke_mask_pedestrian_data(pedestrian_zones):
    # Add pedestrian zones
    pz_shapes = [p.shape for p in pedestrian_zones]
    pz_gdf: Any = GeoDataFrame(geometry=pz_shapes, crs="EPSG:4326")

    return pz_gdf.to_wkt()


def smoke_mask_public_place_data(public_places):
    # TODO pjordan: We should also take into consideration the buildings around it
    # Step 1: Find ground level of this building
    # (might need to be extracted in extract_public_places)
    # Step 2: Find all buildings in a 100m area around it's outline
    # (currently we only have knowledge of its center)
    # Step 3: Mark areas:
    #    a) Mark all areas as "potentially visible"  100m around the outline
    #    b) Mark all areas that have a clear direct viewline as "visible"

    # From what I have found its best to get the topological data using
    # `elevation`. Which downloads a tif file with topological data
    # (Or can extend an existing file, like the one we are creating)
    # We would then need to go over the file again and given all buildings in
    # the perimeter check if some areas are not visible!

    # Simplified approach for now
    # Simply mark everything in a 100m area as no smoke
    pp_shapes = [p.shape for p in public_places]
    pp_gdf: Any = GeoDataFrame(geometry=pp_shapes, crs="EPSG:4326")

    # Convert to meter base as intermediate to make growin easier
    pp_gdf = pp_gdf.to_crs(epsg=32632)
    pp_gdf.geometry = pp_gdf.buffer(100)
    pp_gdf = pp_gdf.to_crs(epsg=4326)

    return pp_gdf.to_wkt()


def create_germany_mask(*, germany_wkt, window, transform, window_transform):
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
    world = np.full((1, window.height, window.width),
                    190, dtype=np.uint8)

    window_transform = wtransform(window, transform)
    smoke_mask = create_smoke_mask(
        no_smoke_wkt=no_smoke_wkt, window=window,
        transform=transform, window_transform=window_transform)
    del no_smoke_wkt

    germany_mask = create_germany_mask(
        germany_wkt=germany_wkt, window=window,
        transform=transform, window_transform=window_transform)
    del germany_wkt

    if germany_mask is not None:
        # Mark germany as green initially
        world[0, germany_mask] = 1
        del germany_mask

    if smoke_mask:
        # Mark probably smoke zones
        world[0, smoke_mask.probably] = 2

        # Mark no smoke zones
        world[0, smoke_mask.forbidden] = 3
        del smoke_mask

    # Write out computed values at correct position
    with write_lock:
        if debug:
            print(f"Writing {window} to file")
        with rasterio.open(output_path, 'r+') as dst:
            dst.write(world, window=window)
        if debug:
            print(f"Finished writing {window} to file")


def create_german_raster(*, out_path,
                         resolution, max_workers,
                         no_smoke_wkt, germany_wkt):
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
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'LZW',
        'blockxsize': 144000,
        'blockysize': 144000,
        'tiled': True,
        'photometric': 'RGBA',
    }

    print(f" |> Creating germany raster ({out_path})")
    # Create a new file with wanted metadata
    with rasterio.open(out_path, 'w', **metadata) as dst:
        # Extract required windows
        windows = [window for _, window in dst.block_windows(1)]

        dst.write_colormap(1, {
            0: (0, 0, 0, 0),
            1: (0, 255, 0, 255),
            2: (255, 255, 0, 255),
            3: (255, 0, 0, 255)
        })

    gc.collect()

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

    print(f" |> Creating world raster ({out_path})")
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


def create_tiles(tif_path, out_dir,
                 *, zoom="0-3", max_workers=8, no_data=None):
    gdal2tiles.generate_tiles(tif_path, out_dir,
                              resume=False, profile="mercator",
                              resampling='average', kml=False,
                              srcnodata=no_data, zoom=zoom,
                              nb_processes=max_workers, )


# Example usage
if __name__ == "__main__":
    # Should be around the number of available cores
    MAX_WORKERS = 4

    # Configure what files should be created
    _recover_state = True
    _create_tifs = True
    _create_tiles = True

    _create_world = True
    _create_germany_public_places = True
    _create_germany_pedestrian_zones = True

    location = "Germany, Baden-Württemberg"

    # Make sure an output folder exists
    Path("output/").mkdir(exist_ok=True)

    if _create_tifs:
        if _recover_state and Path(f"state/{location}").exists():
            no_smoke_public_place_wkt = pd.read_pickle(f"state/{location}/public_place.wkt")
            no_smoke_pedestrian_wkt = pd.read_pickle(f"state/{location}/pedestrian.wkt")
            germany_wkt = Path(f"state/{location}/germany.wkt").read_text()
        else:
            public_places = extract_public_places(location)
            pedestrian_zones = extract_pedestrian_zones(location)

            print(" |> Extracting mask data")
            no_smoke_public_place_wkt = smoke_mask_public_place_data(public_places)
            no_smoke_pedestrian_wkt = smoke_mask_pedestrian_data(public_places)
            germany_wkt = germany_mask_data()

            print(" |> Dumping mask data")
            Path(f"state/{location}").mkdir(parents=True, exist_ok=True)
            no_smoke_public_place_wkt.to_pickle(f"state/{location}/public_place.wkt")
            no_smoke_pedestrian_wkt.to_pickle(f"state/{location}/pedestrian.wkt")
            Path(f"state/{location}/germany.wkt").write_text(germany_wkt)

        print(" |> Creating rasters...")
        if _create_world:
            create_world_raster(width=1440,
                                height=720,
                                germany_wkt=germany_wkt,
                                out_path="output/world_map.tif")
        if _create_germany_public_places:
            create_german_raster(resolution=0.000001,
                                 max_workers=MAX_WORKERS,
                                 no_smoke_wkt=no_smoke_public_place_wkt,
                                 germany_wkt=germany_wkt,
                                 out_path="output/germany_map_public_places.tif")
        if _create_germany_pedestrian_zones:
            # NOTE pjordan: We still need way more precision...
            # tif might not be the best format for this tbh 😅
            # I will have to look into some possible alternatives that are
            # vector based (don't know of any so far)
            # NOTE pjordan: Alternatively if tif is the only option
            # we might need to split germany into sectors or go over things
            # on a per state basis, or similiar
            # -> Apparently we can store everything as shape files,
            #    I however still have to figure out how one would then use that
            #    to create tiles from
            create_german_raster(resolution=0.000001,
                                 max_workers=MAX_WORKERS,
                                 no_smoke_wkt=no_smoke_pedestrian_wkt,
                                 germany_wkt=germany_wkt,
                                 out_path="output/germany_map_pedestrian_zones.tif")

    if _create_tiles:
        print(" |> Creating tiles...")
        if _create_world:
            print(" |> Creating world tiles...")
            create_tiles("output/world_map.tif", "output/world_map/",
                         zoom="0-19", no_data="1")
        if _create_germany_public_places:
            print(" |> Creating germany tiles...")
            create_tiles("output/germany_map_public_places.tif",
                         "output/germany_map_public_places/",
                         zoom="0-19", max_workers=MAX_WORKERS)
        if _create_germany_pedestrian_zones:
            print(" |> Creating germany tiles...")
            create_tiles("output/germany_map_pedestrian_zones.tif",
                         "output/germany_map_pedestrian_zones/",
                         zoom="0-19", max_workers=MAX_WORKERS)
