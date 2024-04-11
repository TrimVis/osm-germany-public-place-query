import osmnx as ox
from pprint import pprint
from dataclasses import dataclass
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString
from enum import Enum

# Configure osmx so it doesn't complain about area size
ox.settings.max_query_area_size = 25000000000
ox.settings.use_cache = True


class Institution(Enum):
    playground = "leisure:playground"
    social_facility = "amenity:social_facility"
    community_centre = "amenity:community_centre"
    social_centre = "amenity:social_centre"
    kindergarten = "amenity:kindergarten"
    childcare = "amenity:childcare"
    school = "amenity:school"
    college = "amenity:college"


INSTITUTION_KEYS = list({
    member.value.split(":")[0] for member in Institution.__members__.values()})
INSTITUTIONS = [
    member.value for member in Institution.__members__.values()]

ShapeType = Point | Polygon | MultiPolygon | LineString | MultiLineString


@dataclass(frozen=True)
class PublicPlace:
    institution: Institution
    name: str | None
    shape: ShapeType


def extract_public_places(place_name, institutions=INSTITUTIONS):
    print(" |> Extracting public places")
    data = []

    for inst in institutions:
        (c, i) = inst.split(":")
        # Query OpenStreetMap for amenities in the specified place
        try:
            g = ox.features_from_place(
                place_name, tags={c: i}
            )
        except ox._errors.InsufficientResponseError:
            print(f" |> \t Found no features for {inst}. Skipping...")
            continue

        print(f" |> \t Found {len(g)} features for {inst}")

        for _, attr in g.iterrows():
            # Detect the institution kind
            ikey = None
            for k in INSTITUTION_KEYS:
                v = str(attr.get(k))
                temp = f"{k}:{v}"
                if temp in INSTITUTIONS:
                    ikey = temp
                    break

            if ikey is None:
                print("Received unexpected institution. Skipping")
                exit(1)

            institution = Institution(ikey)

            name = attr.get('name', None)
            if not isinstance(name, str):
                name = None

            # Append the school's name and geometry
            data.append(
                PublicPlace(
                    institution=institution, name=name, shape=attr.geometry
                ))

    return data


# Example usage
if __name__ == "__main__":
    location = "Waldshut, Baden-Wuerttemberg, Germany"
    pp = extract_public_places(location)
    # Print the first few rows of the DataFrame to check the output
    pprint(pp[:10])
