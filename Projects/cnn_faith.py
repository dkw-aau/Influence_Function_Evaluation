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
from sklearn.metrics.pairwise import cosine_similarity


embeds = load_inception_embeds()

X_train = torch.tensor(embeds["X_train"])
Y_train = torch.tensor(embeds["Y_train"])

X_test = torch.tensor(embeds["X_test"])
Y_test = torch.tensor(embeds["Y_test"])

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
def df_construct(test_idx, train_idxs):
    influences = module.influences(train_idxs=train_idxs, test_idxs=[test_idx])
    
    similarity=[cosine_similarity(X_test[test_idx].numpy().reshape(1,-1), X_train[i].numpy().reshape(1, -1)).item()
               for i in range(len(X_train))]
    squared_diff = (clf(X_train.to(DEVICE)) - Y_train.to(DEVICE))**2

    # Calculate the RMS error for each training point
    train_losses = torch.sqrt(squared_diff)

    # Detach the tensor from the computation graph
    train_losses = train_losses.detach().requires_grad_(False)
    relatif=influences/train_losses

    data = {'Influence': influences.reshape(-1).tolist(), "Relatif":relatif, 'Similarity': similarity, 'Y_train':Y_train.tolist(), 
            'X_train':X_train.numpy().tolist()}
    df = pd.DataFrame(data)
    return df

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_explanation(i):
    
        df = df_construct(i, train_idxs)
        df_pos_sl, df_pos_ol = input_data(df, i, Y_test, sett='positive')
        df_neg_ol, df_neg_sl = input_data(df, i, Y_test, sett='negative')
        selected_indices_pos_sl = greedy_subset_selection(df_pos_sl, N=5, sett='positive', label='same')
        selected_indices_pos_ol = greedy_subset_selection(df_pos_ol, N=5, sett='positive', label='opposite')
        selected_indices_neg_sl = greedy_subset_selection(df_neg_sl, N=5, sett='negative', label='same')
        selected_indices_neg_ol = greedy_subset_selection(df_neg_ol, N=5, sett='negative', label='opposite')
        a=[df_pos_sl.Influence.index[k] for k in selected_indices_pos_sl]
        b=[df_neg_ol.Influence.index[k] for k in selected_indices_neg_ol]
#         c=[df_neg_sl.Influence.index[k] for k in selected_indices_neg_sl]
#         d=[df_pos_ol.Influence.index[k] for k in selected_indices_pos_ol]
        return a+b

from scipy.optimize import linear_sum_assignment

def find_best_matches(embeddings_list1, embeddings_list2):
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_list1, embeddings_list2)

    # Use the Hungarian algorithm to find the optimal assignment
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)

    # Extract the pairs of best matches and their corresponding similarity scores
    best_matches = [(row_indices[i], col_indices[i]) for i in range(len(row_indices))]
    similarity_scores = [-similarity_matrix[row][col] for row, col in best_matches]

    return best_matches, similarity_scores

def average_mbm_similarity(embeddings_list1, embeddings_list2):
    # Find the best matches and their similarity scores
    best_matches, similarity_scores = find_best_matches(embeddings_list1, embeddings_list2)

    # Compute the average cosine similarity using the similarity scores
    avg_similarity = np.mean(np.abs(similarity_scores))
    
    return avg_similarity
    
    
def aide_eval(test_idx):
    simlist=np.array([cosine_similarity(X_test[test_idx].numpy().reshape(1,-1), X_test[i].numpy().reshape(1, -1)).item()
               for i in range(len(X_test))])
    mostsim=simlist.argsort()[-10:]
    leastsim=simlist.argsort()[:10]
    concatenated_array = np.concatenate((mostsim, leastsim)).tolist()
    set1=get_explanation(test_idx)
#     print(f'set1 for {test_idx},  {set1}')
#     mbm_sim=[]
    carray=[]
    fuzzy_jac=[]
    for i in tqdm(concatenated_array):
        influences = module.influences(train_idxs=train_idxs, test_idxs=[i])
        if np.count_nonzero(influences)>800:
            set2=get_explanation(i)
#             jac_sim.append(jaccard_similarity(set(set1), set(set2)))
            similarity_matrix = cosine_similarity(X_train[[set1]], X_train[[set2]])
            mbm=average_mbm_similarity(X_train[[set1]], X_train[[set2]])
#             mbm_sim.append(mbm)
            fuzzy_jac.append(mbm/(len(set2)/5 - mbm))
            carray.append(i)
        else:
            continue
    cosine_sim=simlist[[carray]]
    return cosine_sim.flatten().tolist(),fuzzy_jac


sample_idx = random.sample(range(0, X_test.shape[0]), 50)
cosine_total=[]
# jaccard_total=[]
fuzzy_total=[]
for i in tqdm(sample_idx):
    influences = module.influences(train_idxs=train_idxs, test_idxs=[i])
    if np.count_nonzero(influences)>800:        
        cosine_sim, fuzzy_jac = aide_eval(i)
        cosine_total.append(cosine_sim)
#         mbm_total.append(mbm_sim)
        #     mean_total.append(mean_ex_similarity)
        fuzzy_total.append(fuzzy_jac)
    else:
        continue
    
def flatten_sum(matrix):
    return sum(matrix, [])

# plt.figure()
# plt.scatter(flatten_sum(cosine_total), flatten_sum(mbm_total), alpha=0.5)
# plt.title("AIDE Faithfulness")
# plt.xlabel('Cosine Similarity of Images')
# plt.ylabel('Maximum Bipartite Matching Similarity of Explanations')
# plt.legend(fontsize=7)
# plt.savefig('aide_img_mbm_vs_cos.eps', format='eps')

plt.figure()
plt.scatter(flatten_sum(cosine_total), flatten_sum(fuzzy_total), alpha=0.5)
plt.title("AIDE Faithfulness")
plt.xlabel('Cosine Similarity of Images')
plt.ylabel('Fuzzy Jaccard Similarity of Explanations')
plt.legend(fontsize=7)
plt.savefig('aide_img_fuzzyjac_vs_cos.eps', format='eps')