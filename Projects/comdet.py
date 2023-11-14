import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from sklearn.cluster import SpectralClustering
from image_explainer import *
from utils import *
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.manifold import TSNE
from modules import *
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
embeds = load_inception_embeds()
X_train = torch.tensor(embeds["X_train"])
Y_train = torch.tensor(embeds["Y_train"])

X_test =  torch.cat((torch.tensor(embeds["X_test"]), torch.tensor(embeds["X_train"])), dim=0)
Y_test = torch.cat((torch.tensor(embeds["Y_test"]), torch.tensor(embeds["Y_train"])), dim=0)

# X_test = torch.tensor(embeds["X_test"])
# Y_test = torch.tensor(embeds["Y_test"])

train_set = data.TensorDataset(X_train, Y_train)
test_set = data.TensorDataset(X_test, Y_test)
clf = fit_model(X_train, Y_train)
G = nx.read_gpickle('bipartite_graph.pkl')
points_T = [i for i in range(len(Y_test))]

# Filter nodes and get the biadjacency matrix for points_T
points_T_nodes = [node for node in G.nodes if node in points_T]
biadjacency_mat = bipartite.biadjacency_matrix(G, row_order=points_T_nodes)

# Convert to a regular adjacency matrix
adjacency_mat = biadjacency_mat @ biadjacency_mat.T

# Calculate cosine similarity between node attributes for points_T
node_attributes = {node: X_test[node].numpy() for node in G.nodes if node in points_T_nodes}
attribute_matrix = np.array([node_attributes[node] for node in points_T_nodes])
cosine_sim_matrix = cosine_similarity(attribute_matrix, attribute_matrix)

# Combine adjacency matrix and cosine similarity matrix
combined_matrix = adjacency_mat + cosine_sim_matrix

# Perform spectral clustering using scikit-learn
num_communities = 10  # Adjust as needed
spectral = SpectralClustering(n_clusters=num_communities, affinity='precomputed', random_state=42)
communities = spectral.fit_predict(combined_matrix)

df = pd.DataFrame({'Node': points_T_nodes, 'Community': communities}, index=None)

# Save the DataFrame to a CSV file
df.to_csv('communities.csv', index=False)

# Print the communities for points_T
# for t, community in zip(points_T_nodes, communities):
#     print(f"Community for node {t}: {community}")