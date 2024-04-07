import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import osmnx as ox
import gdal2tiles

ox.settings.use_cache = True
ox.settings.log_console = False


def get_germany_shape():
    # Download the boundary of Germany
    # germany_gdf = ox.geocode_to_gdf('R51477', by_osmid=True)
    germany_gdf = ox.geocode_to_gdf('Germany')

    # return the union of the enclosed points
    return germany_gdf.unary_union


def create_world_raster(germany_shape, output_path='world_map.tif'):
    # Simplified raster dimensions
    width, height = 72000, 36000
    degree = 360 / width
    # World coverage with a simple pixel degree resolution
    transform = from_origin(-180, 90, degree, degree)

    # Raster metadata
    metadata = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'deflate'
    }

    with rasterio.open(output_path, 'w', **metadata) as dst:

        # Initialize the raster to grey
        grey_world = np.full((height, width), fill_value=150, dtype=np.uint8)

        # Create a mask for Germany
        germany_mask = geometry_mask([germany_shape], invert=True, transform=transform,
                                     out_shape=(height, width), all_touched=True)

        # Apply the mask to set Germany area to white (or any other value to not grey out)
        grey_world[germany_mask] = 255  # Highlighting Germany

        dst.write(grey_world, indexes=1)


def create_tiles(tif_path='world_map.tif', out_dir="tiles/"):
    gdal2tiles.generate_tiles(
        tif_path, out_dir, resume=True, zoom="0-7", profile="mercator",
        kml=False, nb_processes=8, resampling='cubic')


if __name__ == "__main__":
    germany_shape = get_germany_shape()
    create_world_raster(germany_shape)
    create_tiles()
