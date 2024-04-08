import osmnx as ox
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from geopandas import GeoDataFrame
from dataclasses import dataclass
from typing import Any
from tqdm import tqdm
import gdal2tiles

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

    def callback(window, transform_fn):
        theight, twidth = (window.height, window.width)
        if debug:
            print("Smoke Mask Callback called")
        # TODO pjordan: Cut with visible area

        no_smoke_mask = geometry_mask(no_smoke_gdf.geometry, transform=transform_fn,
                                      invert=True, out_shape=(theight, twidth),
                                      all_touched=True)
        probably_smoke_mask = np.zeros((theight, twidth), dtype=np.bool_)

        return SmokeMask(no_smoke_mask, probably_smoke_mask)

    print("Generated smoke_mask_callback")
    return callback


def germany_mask_callback():
    germany_shape = get_germany_shape()

    def callback(window, transform_fn):
        theight, twidth = (window.height, window.width)
        if debug:
            print("Germany Mask Callback called")
        return geometry_mask([germany_shape], invert=True,
                             transform=transform_fn,
                             out_shape=(theight, twidth),
                             all_touched=True)

    print("Generated germany_mask_callback")
    return callback


def create_world_raster(public_places, output_path='smoke_map.tif'):
    # Simplified raster dimensions
    width, height = 720000, 360000
    degree = 360 / width
    # World coverage with a simple pixel degree resolution
    # Raster metadata
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': from_origin(-180, 90, degree, degree),
        'compress': 'deflate',
        'blockxsize': height/10,
        'blockysize': width/100,
        'tiled': True
    }

    with rasterio.open(output_path, 'w', **metadata) as dst:
        create_smoke_mask = smoke_mask_callback(public_places)
        get_germany_mask = germany_mask_callback()
        colormap = {
            0: (0, 0, 0, 200),
            1: (0, 255, 0, 100),
            2: (255, 255, 0, 100),
            3: (255, 0, 0, 100),
        }

        # Iterate over the raster in windows
        for ji, window in tqdm(dst.block_windows(1)):
            # Initialize the raster with 0
            if debug:
                print(f"Analyzing Block: {window}")

            # 0 = grey
            # 1 = green
            # 2 = yellow
            # 3 = red
            world = np.zeros((1, window.height, window.width), dtype=np.uint8)

            smoke_mask = create_smoke_mask(
                window=window, transform_fn=dst.window_transform(window))
            germany_mask = get_germany_mask(
                window=window, transform_fn=dst.window_transform(window))

            # Mark germany as green initially
            world[0, germany_mask] = 1

            # Mark probably smoke zones
            world[0, smoke_mask.probably] = 2
            # Mark no smoke zones
            world[0, smoke_mask.forbidden] = 3

            # Make everything darker except for germany
            # world[0, ~germany_mask] = 4

            dst.write(world, window=window)
        dst.write_colormap(1, colormap)


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
