import osmnx as ox

# Disable type hints and therefore errors for ox library
from typing import Any
ox: Any = ox

debug = False

# Configure osmx so it doesn't complain about area size
ox.settings.max_query_area_size = 25000000000
ox.settings.use_cache = True
ox.settings.log_console = debug
