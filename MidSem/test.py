import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('airport_CnToCn_ajc.csv', index_col=0)
G1 = nx.from_pandas_adjacency(df, create_using=nx.DiGraph)

# Calculate indegree and outdegree
in_degrees = dict(G1.in_degree())
out_degrees = dict(G1.out_degree())

# Plot indegree distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
in_deg_values = list(in_degrees.values())
in_deg_hist, in_deg_bins = np.histogram(in_deg_values, bins='auto')  # Change 'auto' to other binning methods if desired
in_deg_bin_centers = (in_deg_bins[:-1] + in_deg_bins[1:]) / 2
plt.loglog(in_deg_bin_centers, in_deg_hist, marker='o', color='b', linestyle='', label='In-degree')
plt.xlabel('Log(Degree)')
plt.ylabel('Log(Frequency)')
plt.title('Indegree Distribution (log-log)')
plt.grid(True)

# Plot outdegree distribution
plt.subplot(1, 2, 2)
out_deg_values = list(out_degrees.values())
out_deg_hist, out_deg_bins = np.histogram(out_deg_values, bins='auto')  # Change 'auto' to other binning methods if desired
out_deg_bin_centers = (out_deg_bins[:-1] + out_deg_bins[1:]) / 2
plt.loglog(out_deg_bin_centers, out_deg_hist, marker='x', color='r', linestyle='', label='Out-degree')
plt.xlabel('Log(Degree)')
plt.ylabel('Log(Frequency)')
plt.title('Outdegree Distribution (log-log)')
plt.grid(True)
plt.tight_layout()
plt.show()
