import osmnx as ox
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.windows import bounds, transform as wtransform
from geopandas import GeoDataFrame
from dataclasses import dataclass
import shapely
from typing import Any
from tqdm import tqdm
import gdal2tiles
import multiprocessing
import concurrent.futures

from extract_public_places import extract_public_places
from create_german_layer import get_germany_shape

# Configure osmx so it doesn't complain about area size
ox.settings.max_query_area_size = 25000000000
ox.settings.use_cache = True
debug = False


@dataclass(frozen=True)
class SmokeMask:
    forbidden: Any
    probably: Any


# TODO pjordan: Process everything in chunks

def smoke_mask_callback(public_places):
    # TODO pjordan: We should also take into consideration the buildings around it
    # Step 1: Find ground level of this building
    # (might need to be extracted in extract_public_places)
    # Step 2: Find all buildings in a 100m area around it's outline
    # (currently we only have knowledge of its center)
    # Step 3: Mark areas:
    #    a) Mark all areas as "potentially visible"  100m around the outline
    #    b) Mark all areas that have a clear direct viewline as "visible"

    # NOTE pjordan: In case we are windowing this is should be precomputed once and not every time

    # Simplified approach for now
    # Simply mark everything in a 100m area as no smoke
    shapes = [p.shape for p in public_places]
    no_smoke_gdf = GeoDataFrame(geometry=shapes, crs="EPSG:4326")
    no_smoke_gdf = no_smoke_gdf.to_crs(epsg=32632)
    no_smoke_gdf.geometry = no_smoke_gdf.buffer(100)
    no_smoke_gdf = no_smoke_gdf.to_crs(epsg=4326)

    def callback(window, transform, window_transform):
        theight, twidth = (window.height, window.width)
        if debug:
            print("Smoke Mask Callback called")

        if shapely.box(*bounds(window, transform)) \
                .intersects(no_smoke_gdf.geometry.any()):
            no_smoke_mask = geometry_mask(no_smoke_gdf.geometry,
                                          transform=window_transform,
                                          out_shape=(theight, twidth),
                                          invert=True,
                                          all_touched=True)
            probably_smoke_mask = np.zeros((theight, twidth), dtype=np.bool_)

            return SmokeMask(no_smoke_mask, probably_smoke_mask)
        return None

    print("Generated smoke_mask_callback")
    return callback


def germany_mask_callback():
    germany_shape = get_germany_shape()

    def callback(window, transform, window_transform):
        theight, twidth = (window.height, window.width)

        if debug:
            print("Germany Mask Callback called")

        # NOTE pjordan: As we do know some the current coordinates we could significantly
        # speed this up, by calculating the expected positions first and checking if we could
        # be in germany... But let's just wait for half an hour for now 😂
        # if True:
        if shapely.box(*bounds(window, transform)) \
                .intersects(germany_shape):
            return geometry_mask([germany_shape], invert=True,
                                 transform=window_transform,
                                 out_shape=(theight, twidth),
                                 all_touched=True)
        return None

    print("Generated germany_mask_callback")
    return callback


def compute_window(output_path, write_lock,
                   window, transform,
                   create_smoke_mask, get_germany_mask):
    # Compute all values
    world = np.full((4, window.height, window.width),
                    190, dtype=np.uint8)

    window_transform = wtransform(window, transform)
    smoke_mask = create_smoke_mask(
        window=window, transform=transform, window_transform=window_transform)
    germany_mask = get_germany_mask(
        window=window, transform=transform, window_transform=window_transform)

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

    # Make everything darker except for germany
    # world[0, ~germany_mask] = 4

    # Write out computed values at correct position
    with write_lock:
        print(f"{window}: Writing to file")
        with rasterio.open(output_path, 'r+') as dst:
            dst.write(world, window=window)
        print(f"{window}: Finished writing to file")


def create_world_raster(public_places, output_path='smoke_map.tif'):
    # Simplified raster dimensions
    w_blocks, h_blocks = 4, 2
    width, height = 14400, 7200
    degree = 360 / width
    transform = from_origin(-180, 90, degree, degree)

    # World coverage with a simple pixel degree resolution
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
        'blockxsize': height/h_blocks,
        'blockysize': width/w_blocks,
        'tiled': True,
        'photometric': 'RGB',
    }

    with rasterio.open(output_path, 'w', **metadata) as dst:
        # Extract required windows
        windows = [window for _, window in dst.block_windows()]

        # # write out colormap
        # colormap = {
        #     0: (0, 0, 0, 200),
        #     6**1: (0, 255, 0, 100),
        #     6**2: (255, 255, 0, 100),
        #     6**3: (255, 0, 0, 100),
        # }
        # dst.write_colormap(1, colormap)

    # write the actual tif file content across multiple processes
    with tqdm(total=len(windows)) as pbar:
        create_smoke_mask = smoke_mask_callback(public_places)
        get_germany_mask = germany_mask_callback()

        with multiprocessing.Manager() as man:
            write_lock = man.Lock()
            # for window in windows:
            #     compute_window(
            #         output_path, write_lock,
            #         window, transform,
            #         create_smoke_mask, get_germany_mask
            #     )
            #     pbar.update(1)
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) \
                    as executor:
                futures = {
                    executor.submit(
                        compute_window,
                        output_path, write_lock,
                        window, transform,
                        create_smoke_mask, get_germany_mask
                    ) for window in windows
                }
                for _ in concurrent.futures.as_completed(futures):
                    pbar.update(1)


def create_tiles(tif_path='smoke_map.tif', out_dir="smoke_tiles/"):
    gdal2tiles.generate_tiles(
        tif_path, out_dir, resume=False, zoom="0-3", profile="mercator",
        kml=False, nb_processes=8, resampling='average')


# Example usage
if __name__ == "__main__":
    location = "Waldshut, Baden-Wuerttemberg, Germany"
    public_places = extract_public_places(location)
    create_world_raster(public_places)
    # create_tiles()
