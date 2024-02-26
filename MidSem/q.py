import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.geocoders import Nominatim

# Load data
flows_df = pd.read_csv('trade-2021-flows.csv', delimiter='\t')
countries_df = pd.read_csv('trade-countries.csv', delimiter='\t')

# Create a dictionary to map country codes to country names
country_names = dict(zip(countries_df['code'], countries_df['name']))

# Create a directed graph
G = nx.DiGraph()

# Add nodes (countries) to the graph
G.add_nodes_from(countries_df['code'])

# Add edges (trade flows) to the graph
for index, row in flows_df.iterrows():
    source = row['from']
    target = row['to']
    amount = row['amount']
    if source in country_names and target in country_names:
        G.add_edge(source, target, weight=amount)

# Calculate node degrees
node_degrees = dict(G.degree)

# Calculate edge weights
edge_weights = [d['weight'] for u, v, d in G.edges(data=True)]

# Get coordinates for countries
geolocator = Nominatim(user_agent="networks")
country_locations = {}

for country_code in countries_df['code']:
    country_name = country_names.get(country_code, "")
    if country_name:
        location = geolocator.geocode(country_name)
        if location:
            country_locations[country_code] = (location.longitude, location.latitude)

# Create a world map
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -90, 90])

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# Plot the graph on the map
pos = country_locations
nx.draw(G, pos, ax=ax, with_labels=True, node_size=[node_degrees[node] * 10 for node in G.nodes()],
        node_color='r', font_size=8, font_weight='bold', edge_color='gray',
        width=[float(w) / max(edge_weights) * 5 for w in edge_weights])

plt.title('International Trade Network (2021)')
plt.show()
