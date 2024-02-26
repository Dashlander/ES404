import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Load data
adjacency_matrix_df = pd.read_csv('airport_CnToCn_ajc.csv', index_col=0)

# Create a directed graph from the adjacency matrix
G = nx.from_pandas_adjacency(adjacency_matrix_df, create_using=nx.DiGraph)

# Calculate node degrees
node_degrees = dict(G.degree)

# Calculate edge weights
edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]

# Get coordinates for countries with caching
geolocator = Nominatim(user_agent="network")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
country_locations = {}

for country_name in adjacency_matrix_df.index:
    location = geocode(country_name)
    if location:
        country_locations[country_name] = (location.longitude, location.latitude)

# Create a world map
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -90, 90])

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# Plot the graph on the map
pos = {country_name: (lon, lat) for country_name, (lon, lat) in country_locations.items()}
nx.draw(G, pos, ax=ax, with_labels=True, node_size=[node_degrees[node] * 5 for node in G.nodes()], node_color='b', font_size=8, edge_color='black', width = [float(w) / max(edge_weights) * 5 for w in edge_weights])

plt.title('International Aviation Network')
plt.show()
