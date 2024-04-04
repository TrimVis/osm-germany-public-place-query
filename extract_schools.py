import osmnx as ox
from pprint import pprint
from dataclasses import dataclass
from shapely.geometry import Point, Polygon, LineString, MultiPolygon, MultiLineString
from enum import Enum

# Configure osmx so it doesn't complain about area size
ox.settings.max_query_area_size = 25000000000


class Institution(Enum):
    school = "school"
    college = "college"


@dataclass(frozen=True)
class PublicPlace:
    institution: Institution
    name: str | None
    lon: float
    lat: float


def extract_public_places(place_name, amenities):
    data = []

    for amenity in amenities:
        # Query OpenStreetMap for amenities in the specified place
        g = ox.features_from_place(
            place_name, tags={"amenity": amenity}
        )

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


# Example usage
if __name__ == "__main__":
    location = "Baden-Wuerttemberg, Germany"
    amenities = ["school", "college"]
    schools_df = extract_public_places(location, amenities)
    # Print the first few rows of the DataFrame to check the output
    pprint(schools_df[:10])
