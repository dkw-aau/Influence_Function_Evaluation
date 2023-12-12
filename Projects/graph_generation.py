from image_explainer import *
from utils2 import *
import numpy as np
import pandas as pd
import torch
import torchvision
from modules import *
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from networkx.algorithms import bipartite
from sklearn.cluster import SpectralClustering

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

module = LiSSAInfluenceModule(
    model=clf,
    objective=BinClassObjective(),
    train_loader=data.DataLoader(train_set, batch_size=32),
    test_loader=data.DataLoader(test_set, batch_size=32),
    device=DEVICE,
    damp=0.001,
    repeat= 1,
    depth=1800,
    scale= 10,
)
train_idxs = list(range(X_train.shape[0]))


# Function to generate Es and values for a given T
def df_construct(test_idx, train_idxs):
    influences = module.influences(train_idxs=train_idxs, test_idxs=[test_idx])
    
    similarity=[cosine_similarity(X_test[test_idx].numpy().reshape(1,-1), X_train[i].numpy().reshape(1, -1)).item()
               for i in range(len(X_train))]

    data = {'Influence': influences.reshape(-1).tolist(),'Similarity': similarity, 'Y_train':Y_train.tolist(), 'X_train':X_train.numpy().tolist()}
    df = pd.DataFrame(data)
    return df
def get_explanation(i):
    
        df = df_construct(i, train_idxs)
        df_pos_sl = input_data(df, i, Y_test)
        selected_indices_pos_sl = greedy_subset_selection(df_pos_sl, N=5)
        a={df_pos_sl.Influence.index[k]:df_pos_sl.Influence.tolist()[k] for k in selected_indices_pos_sl}
        return a
    
def generate_influential_samples(t, aide=False):
    if aide or module.influences(train_idxs=train_idxs, test_idxs=[t]).sum().item()==0:
        return get_explanation(t)
    else:
        scaler = MinMaxScaler()
        inf_list=scaler.fit_transform(module.influences(train_idxs=train_idxs, test_idxs=[t]).reshape(-1, 1))
        inf_list=torch.tensor(inf_list.squeeze()).sort(descending=True)
        return {inf_list[1][i].item(): inf_list[0][i].item() for i in range(5)}

# List of T's
points_T = [i for i in range(len(Y_test))]

# Create influential_samples dictionary automatically
influential_samples = {t: generate_influential_samples(t, aide=True) for t in tqdm(points_T)}


# Create a bipartite graph
G = nx.Graph()

# Create a bipartite graph
G = nx.Graph()

# Add nodes from sets T and E with vector representations and image tensors
for i in points_T:
    G.add_node(i, bipartite=0, embedding=X_test[i].numpy())

for point, samples in influential_samples.items():
    for sample, score in samples.items():
        G.add_node(f'ex-{sample}', bipartite=1, embedding=X_train[sample].numpy())
        G.add_edge(point, f'ex-{sample}', weight=score)

# Save the graph as data
nx.write_gpickle(G, '/data/ikhtiyor/bipartite_graph_5.pkl')


# # Filter nodes and get the biadjacency matrix for points_T
# biadjacency_mat = bipartite.biadjacency_matrix(G, row_order=points_T)

# # Convert to a regular adjacency matrix
# adjacency_mat = biadjacency_mat @ biadjacency_mat.T

# # Calculate cosine similarity between node attributes for points_T
# attribute_matrix = np.array([G.nodes[node]['embedding'] for node in points_T])
# cosine_sim_matrix = cosine_similarity(attribute_matrix, attribute_matrix)

# # Combine adjacency matrix and cosine similarity matrix
# combined_matrix = adjacency_mat + cosine_sim_matrix

# # Perform spectral clustering using scikit-learn
# num_communities = 10  # Adjust as needed
# spectral = SpectralClustering(n_clusters=num_communities, affinity='precomputed', random_state=42, n_jobs=-1)
# communities = spectral.fit_predict(combined_matrix)

# df = pd.DataFrame({'Node': points_T, 'Community': communities, 'Label':Y_test}, index=None)

# Save the DataFrame to a CSV file
# df.to_csv('/data/ikhtiyor/communities_aide2.csv', index=False)