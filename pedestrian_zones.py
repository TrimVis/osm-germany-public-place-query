import osmnx as ox
from pprint import pprint
from dataclasses import dataclass
from shapely.geometry import (
    Point, Polygon, LineString, MultiPolygon, MultiLineString
)
from enum import Enum
from config import debug

# Disable type hints and therefore errors for ox library
from typing import Any
ox: Any = ox


class Zone(Enum):
    pedestrian = "highway:pedestrian"


ZONE_KEYS = list({
    member.value.split(":")[0] for member in Zone.__members__.values()})
ZONES = [
    member.value for member in Zone.__members__.values()]


ShapeType = Point | Polygon | MultiPolygon | LineString | MultiLineString


@dataclass(frozen=True)
class PedestrianZone:
    zone: Zone
    name: str | None
    shape: ShapeType


def extract_pedestrian_zones(place_name, zones=ZONES):
    print(" |> Extracting pedestrian zones")
    data = []

    for zone in zones:
        if debug:
            print(zone)
        (c, i) = zone.split(":")
        # Query OpenStreetMap for amenities in the specified place
        try:
            g = ox.features_from_place(
                place_name, tags={c: i}
            )
        except ox._errors.InsufficientResponseError:
            print(f" |> \t Found no features for {zone}. Skipping...")
            continue

        print(f" |> \t Found {len(g)} features for {zone}")

        for _, attr in g.iterrows():
            # Detect the institution kind
            ikey = None
            for k in ZONE_KEYS:
                v = str(attr.get(k))
                temp = f"{k}:{v}"
                if temp in ZONES:
                    ikey = temp
                    break

            if ikey is None:
                print("Received unexpected zone. Skipping")
                exit(1)

            zone_ = Zone(ikey)

            name = attr.get('name', None)
            if not isinstance(name, str):
                name = None

            # Append the school's name and geometry
            data.append(
                PedestrianZone(
                    zone=zone_, name=name, shape=attr.geometry
                ))

    return data


# Example usage
if __name__ == "__main__":
    location = "Waldshut, Baden-Wuerttemberg, Germany"
    pp = extract_pedestrian_zones(location)
    # Print the first few rows of the DataFrame to check the output
    pprint(pp[:10])
