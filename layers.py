import osmnx as ox
import numpy as np
import shapely
import shapely.wkt

from rasterio.features import geometry_mask
from rasterio.windows import bounds
from geopandas import GeoDataFrame
from dataclasses import dataclass
from typing import Any


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
