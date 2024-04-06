import osmnx as ox
from pprint import pprint
from dataclasses import dataclass
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString
from enum import Enum

# Configure osmx so it doesn't complain about area size
ox.settings.max_query_area_size = 25000000000


class Institution(Enum):
    social_center = "social_center"
    kindergarten = "kindergarten"
    school = "school"
    college = "college"


INSTITUTIONS = [member.value for member in Institution.__members__.values()]


@dataclass(frozen=True)
class PublicPlace:
    institution: Institution
    name: str | None
    lon: float
    lat: float


def extract_public_places(place_name, amenities=INSTITUTIONS):
    data = []

    for amenity in amenities:
        # Query OpenStreetMap for amenities in the specified place
        try:
            g = ox.features_from_place(
                place_name, tags={"amenity": amenity}
            )
        except ox._errors.InsufficientResponseError:
            print(f"Could not find any features for {amenity}...")
            print("Skipping")
            continue

        print(f"Found {len(g)} features for {amenity}")

        for _, attr in g.iterrows():
            # Detect the institution kind
            institution = Institution(attr['amenity'])

            # Extract the name, if available, else use None
            name = attr.get('name', None)
            if not isinstance(name, str):
                name = None

            # Check the geometry type and extract coordinates accordingly
            lon, lat = None, None
            if isinstance(attr.geometry, Point):
                lon, lat = attr.geometry.x, attr.geometry.y
            elif isinstance(attr.geometry, (Polygon, MultiPolygon, LineString, MultiLineString)):
                lon, lat = attr.geometry.centroid.x, attr.geometry.centroid.y

            # Append the school's name and coordinates to the list
            if lon is not None and lat is not None:
                data.append(
                    PublicPlace(institution=institution,
                                name=name, lon=lon, lat=lat))
            else:
                print("Missing location for institution")

    return data


def find_visible_areas(areas):
    # TODO pjordan: We should also take into consideration the buildings around it
    for area in areas:
        _lon, _lat = area.lon, area.lat

        # Step 1: Find ground level of this building
        # (might need to be extracted in extract_public_places)
        # Step 2: Find all buildings in a 100m area around it's outline
        # (currently we only have knowledge of its center)
        # Step 3: Mark areas:
        #    a) Mark all areas as "potentially visible"  100m around the outline
        #    b) Mark all areas that have a clear direct viewline as "visible"

        pass


# Example usage
if __name__ == "__main__":
    location = "Baden-Wuerttemberg, Germany"
    pp = extract_public_places(location)
    # Print the first few rows of the DataFrame to check the output
    pprint(pp[:10])
