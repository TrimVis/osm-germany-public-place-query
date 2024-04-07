import osmnx as ox
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from geopandas import GeoDataFrame
from dataclasses import dataclass
from typing import Any
import gdal2tiles

from extract_public_places import extract_public_places
from create_german_layer import get_germany_shape

# Configure osmx so it doesn't complain about area size
ox.settings.max_query_area_size = 25000000000
ox.settings.use_cache = True


# Simplified raster dimensions
width, height = 72000, 36000
degree = 360 / width
# World coverage with a simple pixel degree resolution
transform = from_origin(-180, 90, degree, degree)


@dataclass(frozen=True)
class SmokeMask:
    forbidden: Any
    probably: Any


def create_smoke_mask(public_places):

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
    no_smoke_gdf = GeoDataFrame(geometry=shapes, crs="EPSG:4326")
    no_smoke_gdf = no_smoke_gdf.to_crs(epsg=32632)
    no_smoke_gdf['geometry'] = no_smoke_gdf.buffer(100)
    no_smoke_gdf = no_smoke_gdf.to_crs(epsg=4326)

    no_smoke_mask = geometry_mask(no_smoke_gdf.geometry, transform=transform,
                                  out_shape=(height, width), all_touched=True)
    probably_smoke_mask = np.zeros((height, width), dtype=np.bool_)

    return SmokeMask(no_smoke_mask, probably_smoke_mask)


def get_germany_mask():
    germany_shape = get_germany_shape()
    return geometry_mask([germany_shape], invert=True, transform=transform,
                         out_shape=(height, width), all_touched=True)


def create_world_raster(smoke_mask, germany_mask, output_path='smoke_map.tif'):

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
        'photometric': "RGB",
    }

    with rasterio.open(output_path, 'w', **metadata) as dst:

        # Initialize the raster with 0
        world = np.zeros((4, height, width), dtype=np.uint8)


        # Mark everything as green initially
        world[1, :, :] = 255
        world[3, :, :] = 100

        # Mark probably smoke zones (green and red mix to yellow)
        world[0][smoke_mask.probably] = 255
        # Mark no smoke zones
        world[0][smoke_mask.forbidden] = 255
        world[1][smoke_mask.forbidden] = 0

        # Make everything darker except for germany
        world[3][:] = 180
        world[3][germany_mask] = 0
        world[0][~germany_mask] = 0
        world[1][~germany_mask] = 0
        world[2][~germany_mask] = 0

        dst.write(world)


def create_tiles(tif_path='smoke_map.tif', out_dir="smoke_tiles/"):
    gdal2tiles.generate_tiles(
        tif_path, out_dir, resume=False, zoom="0-3", profile="mercator",
        kml=False, nb_processes=8, resampling='average')


# Example usage
if __name__ == "__main__":
    location = "Waldshut, Baden-Wuerttemberg, Germany"
    public_places = extract_public_places(location)
    smoke_mask = create_smoke_mask(public_places)
    germany_mask = get_germany_mask()
    create_world_raster(smoke_mask, germany_mask)
    create_tiles()
