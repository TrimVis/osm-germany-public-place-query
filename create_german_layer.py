import rasterio
import fiona
import rasterio.mask
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import geopandas as gpd
import numpy as np
import gdal2tiles
import matplotlib.pyplot as plt


germany_shapefile = './germany_shapefile/de_1km.shp'
out_tif = './layers/rasters/germany_area.tif'
out_tiles = './layers/tiles/germany_outline/'


def create_tif_test():
    with fiona.open(germany_shapefile, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open("./tmp/world.rgb.tif") as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(out_image)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


def test_tif():
    germany_gdf = gpd.read_file(germany_shapefile)
    print("--- GDF ---")
    print(germany_gdf)
    print("--- Bounds ---")
    print(germany_gdf.bounds)
    print(germany_gdf.bounds.iloc[0])
    print("--- MinX ---")
    print(germany_gdf.bounds.iloc[0].minx)
    print("--- MaxX ---")
    print(germany_gdf.bounds.iloc[0].maxx)
    print("--- MinY ---")
    print(germany_gdf.bounds.iloc[0].miny)
    print("--- MaxY ---")
    print(germany_gdf.bounds.iloc[0].maxy)
    print("--- CRS ---")
    print(germany_gdf.crs)


def create_tif():
    # Load the shapefile of Germany
    print("Reading shape file")
    germany_gdf = gpd.read_file(germany_shapefile)

    # Create a transform and raster dimensions that cover the extent of Germany
    resolution = 1000
    out_shape = (resolution, resolution)
    bounds = germany_gdf.bounds.iloc[0]
    transform = rasterio.transform.from_bounds(
        west=bounds.minx, south=bounds.miny, east=bounds.maxx,
        north=bounds.maxy, width=resolution, height=resolution)

    print("Rasterizing and preparing image")
    # Rasterize the shapefile
    with rasterio.open(out_tif, 'w',
                       driver='GTiff', height=out_shape[0],
                       width=out_shape[1], count=3,
                       dtype='uint8', crs=germany_gdf.crs,
                       transform=transform) as out_raster:
        mask = geometry_mask(germany_gdf.geometry, invert=True,
                             transform=transform, all_touched=True,
                             out_shape=out_shape)

        print(mask)

        # # Paints all areas as see through
        # visible_color = 200
        # base_data = visible_color * \
        #     np.ones((out_shape[0], out_shape[1]), dtype=rasterio.uint8)

        # # Paints non-Germany areas grey
        # grey_color = 150
        # base_data[~mask] = grey_color
        # out_raster.write(base_data, 1)
        # out_raster.write(base_data, 1)

        # Paints all areas as red
        base_data = 255 * \
            np.ones((out_shape[0], out_shape[1]), dtype=rasterio.uint8)
        out_raster.write(base_data, 1)

        # Paints non-Germany areas green
        base_data = np.zeros(
            (out_shape[0], out_shape[1]), dtype=rasterio.uint8)
        base_data[~mask] = 255
        out_raster.write(base_data, 2)

        # Paints strip blue
        base_data = np.zeros(
            (out_shape[0], out_shape[1]), dtype=rasterio.uint8)
        base_data[:][0:100] = 255
        out_raster.write(base_data, 3)


def create_tiles():
    print("Generating tiles")
    gdal2tiles.generate_tiles(
        out_tif, out_tiles, zoom="0-2", profile="mercator", kml=False,
        nb_processes=8, webviewer="none")


if __name__ == "__main__":
    create_tif_test()
    # test_tif()

    # create_tif()
    # create_tiles()
