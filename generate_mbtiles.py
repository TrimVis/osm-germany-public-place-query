import shapely
import shapely.wkt
import pandas as pd
import subprocess

from pathlib import Path
from geopandas import GeoDataFrame

from public_places import extract_public_places
from pedestrian_zones import extract_pedestrian_zones

from layers import (
    germany_mask_data,
    smoke_mask_public_place_data,
    smoke_mask_pedestrian_data,
)


def create_vector(*, out_path, wkt, kind=None):
    gdf = GeoDataFrame(geometry=[shapely.wkt.loads(wkt)])
    del wkt
    json = gdf.to_json()
    print(f" |> Creating {kind + ' ' if kind else ' '}GeoJSON ({out_path})")
    # Create a new file with wanted metadata
    with open(out_path, 'w') as f:
        f.write(json)


def check_tippecanoe():
    print(" |> Checking of tippecanoe is installed")
    result = subprocess.run(["tippecanoe", "-v"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print(" |> Success!")
    else:
        print(" |> Error: Please install 'tippecanoe' to continue!")
        exit(1)


def create_vector_tiles(in_path, out_path, kind=None):
    print(f" |> Creating {kind + ' ' if kind else ' '}MbTiles ({out_path})")
    # TODO pjordan: Figure out what flags to pass here
    result = subprocess.run([
        "tippecanoe",
        "-zg", "--drop-densest-as-needed",
        "--force",
        "-o", out_path,
        in_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print(" |> Success!")
    else:
        print(" |> Error: \n", result.stderr)


# Example usage
if __name__ == "__main__":
    # Should be around the number of available cores
    MAX_WORKERS = 4

    # Configure what files should be created
    _recover_state = True
    _create_vectors = True
    _create_tiles = True

    _create_world = True
    _create_germany_public_places = True
    _create_germany_pedestrian_zones = True

    location = "Germany, Baden-Württemberg"

    # Make sure an output folder exists
    Path("output/").mkdir(exist_ok=True)

    if _create_vectors:
        if _recover_state and Path(f"state/{location}").exists():
            no_smoke_public_place_wkt = pd.read_pickle(
                f"state/{location}/public_place.wkt")
            no_smoke_pedestrian_wkt = pd.read_pickle(
                f"state/{location}/pedestrian.wkt")
            germany_wkt = Path(f"state/{location}/germany.wkt").read_text()
        else:
            public_places = extract_public_places(location)
            pedestrian_zones = extract_pedestrian_zones(location)

            print(" |> Extracting mask data")
            no_smoke_public_place_wkt = smoke_mask_public_place_data(
                public_places)
            no_smoke_pedestrian_wkt = smoke_mask_pedestrian_data(public_places)
            germany_wkt = germany_mask_data()

            print(" |> Dumping mask data")
            Path(f"state/{location}").mkdir(parents=True, exist_ok=True)
            no_smoke_public_place_wkt.to_pickle(
                f"state/{location}/public_place.wkt")
            no_smoke_pedestrian_wkt.to_pickle(
                f"state/{location}/pedestrian.wkt")
            Path(f"state/{location}/germany.wkt").write_text(germany_wkt)

        print(" |> Creating vectors...")
        if _create_world:
            create_vector(wkt=germany_wkt,
                          out_path="output/world_map.geojson",
                          kind="German Border Outline")
        if _create_germany_public_places:
            create_vector(wkt=germany_wkt,
                          out_path="output/public_places.geojson",
                          kind="German Public Places")
        if _create_germany_pedestrian_zones:
            create_vector(wkt=germany_wkt,
                          out_path="output/pedestrian_zones.geojson",
                          kind="German Pedestrian Zones")

    if _create_tiles:
        check_tippecanoe()

        print(" |> Creating tiles...")
        if _create_world:
            print(" |> Creating world tiles...")
            create_vector_tiles("output/world_map.geojson",
                                "output/world_map.mbtiles")
        if _create_germany_public_places:
            print(" |> Creating germany public places tiles...")
            create_vector_tiles("output/public_places.geojson",
                                "output/public_places.mbtiles")
        if _create_germany_pedestrian_zones:
            print(" |> Creating germany pedestrian zone tiles...")
            create_vector_tiles("output/pedestrian_zones.geojson",
                                "output/pedestrian_zones.mbtiles")
