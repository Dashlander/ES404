import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Load data from matrix
adjacency_matrix_df = pd.read_csv('airport_CnToCn_ajc.csv', index_col=0)

# Create a directed graph from the adjacency matrix
G = nx.from_pandas_adjacency(adjacency_matrix_df, create_using=nx.DiGraph)

# Load country coordinates from CSV file
country_coordinates_df = pd.read_csv('test.csv', delimiter='\t', usecols=['latitude', 'longitude', 'name'])

# Filter coordinates for countries present in the adjacency matrix
country_coordinates_df = country_coordinates_df[country_coordinates_df['name'].isin(adjacency_matrix_df.index)]

# Create a dictionary of country coordinates
country_locations = country_coordinates_df.set_index('name')[['longitude', 'latitude']].to_dict(orient='index')

# Create a world map
plt.figure(figsize=(18, 9))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -90, 90])

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)

# Convert coordinates to the projection used by Cartopy
pos_cartopy = {country: (coords['longitude'], coords['latitude']) for country, coords in country_locations.items()}
pos = {country: ax.projection.transform_point(lon, lat, ccrs.PlateCarree()) for country, (lon, lat) in pos_cartopy.items()}

node_degrees = dict(G.degree())
node_sizes = [node_degrees[node] * 5 for node in G.nodes()]

# Compute edge widths based on edge weights
edge_weights = nx.get_edge_attributes(G, 'weight')
max_weight = max(edge_weights.values())
edge_widths = [edge_weights.get(edge, 0) / max_weight * 10 for edge in G.edges()]

# Plot the graph on the map
nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_sizes, node_color='#66c2ff', font_size=5, font_weight='bold', edge_color='#454545', width = edge_widths)

plt.title('Air Connectivity Network')
plt.show()
