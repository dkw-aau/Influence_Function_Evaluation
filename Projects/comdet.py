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
G = nx.read_gpickle('/data/ikhtiyor/bipartite_graph_aide_all_10.pkl')
points_T = [i for i in range(len(Y_test))]

# Filter nodes and get the biadjacency matrix for points_T
points_T_nodes = [node for node in G.nodes if node in points_T]
nodes_bipartite_0 = [node for node in G.nodes if G.nodes[node]['bipartite'] == 0]

# Calculate cosine similarity between attribute vectors of nodes with bipartite=0
attribute_matrix = [G.nodes[node]['embedding'] for node in nodes_bipartite_0]
cosine_sim_matrix = cosine_similarity(attribute_matrix, attribute_matrix)

# Create a graph with nodes from bipartite=0 and weighted edges based on cosine similarity
weighted_projection = nx.Graph()
weighted_projection.add_nodes_from(nodes_bipartite_0)

# Add weighted edges based on cosine similarity scores and the neighbours
for i, node1 in enumerate(nodes_bipartite_0):
    for j, node2 in enumerate(nodes_bipartite_0):
        if i < j:
            if len(set(G.neighbors(node1)).intersection(set(G.neighbors(node2))))!=0:
                for k in set(G.neighbors(node1)).intersection(set(G.neighbors(node2))):
                    
                    if G[i][k]['weight'] + G[j][k]['weight'] ==0:
                        weight = cosine_sim_matrix[i][j]
                    else:
                        
                        weight = cosine_sim_matrix[i][j]+2*G[i][k]['weight']*G[j][k]['weight']/(G[i][k]['weight'] + G[j][k]['weight'])
#             else:
#                 weight = cosine_sim_matrix[i][j]/10
                    weighted_projection.add_edge(node1, node2)

adjacency_mat = nx.adjacency_matrix(weighted_projection).toarray()
# Calculate cosine similarity between node attributes for points_T
# attribute_matrix = np.array([G.nodes[node]["embedding"] for node in points_T])
attribute_matrix = np.asarray([G.nodes[node]["embedding"] for node in points_T])
cosine_sim_matrix = cosine_similarity(attribute_matrix, attribute_matrix)

# Combine adjacency matrix and cosine similarity matrix
combined_matrix = adjacency_mat + cosine_sim_matrix

# Perform spectral clustering using scikit-learn
num_communities = 20 
print('until here')
spectral = SpectralClustering(n_clusters=num_communities, affinity='precomputed', random_state=42, n_jobs=1)

# from sklearn.decomposition import PCA

# Reduce dimensionality using PCA
# pca = PCA(n_components=100)  # Choose an appropriate number of components
# combined_matrix_pca = pca.fit_transform(combined_matrix)


communities = spectral.fit_predict(adjacency_mat)
df = pd.DataFrame({'Node': points_T, 'Community': communities, 'Label':Y_test}, index=None)

# # Save the DataFrame to a CSV file
df.to_csv('communities.csv', index=False)