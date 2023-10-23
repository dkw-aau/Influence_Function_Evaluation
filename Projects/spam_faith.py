import pandas as pd
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils import data
from modules import *
from base import BaseObjective
import torch.nn.functional as F
import random

import matplotlib.pyplot as plt

with open('data/spambert.pkl', 'rb') as f:
    df = pickle.load(f)

# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
df['label'] = df['label'].replace({'spam': 1, 'ham': 0})
X=torch.stack(df.embedding.tolist())
y=df.label

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=2001)

text_train = df.message[Y_train.index].tolist()
X_train = torch.tensor(X_train)
Y_train = torch.tensor(Y_train.tolist()).type('torch.FloatTensor')

text_test = df.message[Y_test.index].tolist()
Y_test = torch.tensor(Y_test.tolist()).type('torch.FloatTensor')
X_test = torch.tensor(X_test)

L2_WEIGHT = 1e-4
def fit_model(X, Y):
    C = 1 / (X.shape[0] * L2_WEIGHT)
    sk_clf = linear_model.LogisticRegression(C=C, tol=1e-8, max_iter=1000)
    sk_clf = sk_clf.fit(X.numpy(), Y.numpy())

    # recreate model in PyTorch
    fc = nn.Linear(768, 1, bias=True)
    fc.weight = nn.Parameter(torch.tensor(sk_clf.coef_))
    fc.bias = nn.Parameter(torch.tensor(sk_clf.intercept_))

    pt_clf = nn.Sequential(
        fc,
        nn.Flatten(start_dim=-2),
        nn.Sigmoid()
    )

    pt_clf = pt_clf.to(device=DEVICE, dtype=torch.float32)
    return pt_clf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_set = data.TensorDataset(X_train, Y_train)
test_set = data.TensorDataset(X_test, Y_test)
clf = fit_model(X_train, Y_train)

class BinClassObjective(BaseObjective):

    def train_outputs(self, model, batch):
        return model(batch[0])

    def train_loss_on_outputs(self, outputs, batch):
        return F.binary_cross_entropy(outputs, batch[1])

    def train_regularization(self, params):
        return L2_WEIGHT * torch.square(params.norm())

    def test_loss(self, model, params, batch):
        outputs = model(batch[0])
        return F.binary_cross_entropy(outputs, batch[1])
    
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
import torch.nn.functional as F
def df_construct(test_idx, train_idxs):
    influences = module.influences(train_idxs=train_idxs, test_idxs=[test_idx])
    similarity=[cosine_similarity(X_test[test_idx].numpy().reshape(1,-1), X_train[i].numpy().reshape(1, -1)).item()
               for i in range(len(X_train))]

    data = {'Influence': influences.reshape(-1).tolist(), 'Similarity': similarity, 'Label':Y_train.tolist(), 
            'X_train':X_train.numpy().tolist(), 'message':text_train}
    df = pd.DataFrame(data)
    return df
    
train_idxs = list(range(X_train.shape[0]))


def input_data(df, test_idx, sett=None):
    
    scaler = MinMaxScaler()
    
    influence_pos = [i for i in df.Influence.tolist() if i>0]
    q3p, q1p = np.percentile(influence_pos, [75 ,25])
    iqrp = q3p - q1p
    influenceIQp=np.array([i for i in influence_pos if i>q3p+3*iqrp])
    n_p=len(influenceIQp)
    
    influence_neg = [i for i in df.Influence.tolist() if i<0]
    q3n, q1n = np.percentile(influence_neg, [75 ,25])
    iqrn = q3n - q1n
    influenceIQn=np.array([i for i in influence_neg if i<q1n-2*iqrn])
    nn=len(influenceIQn)
    
    if sett == 'positive':

        df_pos = df[df.Influence>0].sort_values('Influence', ascending=False) #.reset_index(drop=True)
        df_pos[['Influence', 'Similarity']]= scaler.fit_transform(df_pos[['Influence', 'Similarity']])
        df_pos_sl = df_pos[df_pos.Label==Y_test[test_idx].item()][:n_p]
        df_pos_ol = df_pos[df_pos.Label!=Y_test[test_idx].item()][:n_p]
        
        return df_pos_sl, df_pos_ol
    
    elif sett=='negative':
       
        df_neg = df[df.Influence<0].sort_values('Influence', ascending=True) #.reset_index(drop=True)
        df_neg[['Influence', 'Similarity']]= scaler.fit_transform(df_neg[['Influence', 'Similarity']])
        df_neg_sl = df_neg[df_neg.Label==Y_test[test_idx].item()][:nn]
        df_neg_ol = df_neg[df_neg.Label!=Y_test[test_idx].item()][:nn]
        
        return df_neg_ol, df_neg_sl
    
def calculate_cosine_similarity(a, b):
    """Calculate cosine similarity between two arrays."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def greedy_subset_selection(df, N, sett=None, label=None):
    
    arrays = [np.array(i) for i in df.X_train]
    influence_scores = df.Influence.tolist()  # List of influence scores for each array
    prox = df.Similarity.tolist()
    n_arrays = len(arrays)
    selected_indices = []
    
    wi, ws, wd=0.2, 0.9, 0.3
 
    # Start with the array with the highest influence score
    if sett =='positive' and label=='opposite':
        initial_idx = np.argmax(influence_scores)
    elif sett =='positive' and label=='same':
        initial_idx = np.argmax(influence_scores)
    if sett =='negative' and label=='opposite':
        initial_idx = np.argmax(influence_scores)
    elif sett =='negative' and label=='same':
        initial_idx = np.argmax(influence_scores)
    selected_indices.append(initial_idx)
    selected_array = arrays[initial_idx]
    
    while len(selected_indices) < N:
        max_gain = -np.inf
        selected_idx = None
        
        # Iterate over the remaining arrays
        for i in range(n_arrays):
            if i not in selected_indices:
                current_array = arrays[i]
                final_list = list(map(lambda x: calculate_cosine_similarity(current_array, arrays[x]), selected_indices))
                if any(i>0.97 for i in final_list):
                    continue
                else:
                    
                    diversity = np.mean(final_list)
                    # Calculate combined score of diversity and influence score
                    if sett =='positive' and label=='same':
                        combined_score = 0.4*influence_scores[i]+0.8*prox[i] #-0.3*diversity
#                         combined_score = wi*influence_scores[i] + ws*prox[i] - wd*similarity
                    elif sett =='negative' and label=='opposite':
                        combined_score = -0.4*influence_scores[i]+0.8*prox[i] #- 0.3*diversity                        
                    elif sett =='positive' and label=='opposite':
                        combined_score = 0.4*influence_scores[i]+0.8*prox[i] #-0.3*diversity
#                         combined_score = wi*influence_scores[i] - wd*similarity
                    elif sett =='negative' and label=='same':
                        combined_score = -0.4*influence_scores[i]+0.8*prox[i] #-0.3*diversity
                    # Update selected array if it provides the highest gain
                    if combined_score > max_gain:
                        max_gain = combined_score
                        selected_idx = i
        
        # Add selected array to the subset
        selected_indices.append(selected_idx)
    
    return selected_indices

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_explanation(i):
    
        df = df_construct(i, train_idxs)
        df_pos_sl, df_pos_ol = input_data(df, i, sett='positive')
        df_neg_ol, df_neg_sl = input_data(df, i, sett='negative')
        selected_indices_pos_sl = greedy_subset_selection(df_pos_sl, N=5, sett='positive', label='same')
        selected_indices_pos_ol = greedy_subset_selection(df_pos_ol, N=5, sett='positive', label='opposite')
        selected_indices_neg_sl = greedy_subset_selection(df_neg_sl, N=5, sett='negative', label='same')
        selected_indices_neg_ol = greedy_subset_selection(df_neg_ol, N=5, sett='negative', label='opposite')
        a=[df_pos_sl.Influence.index[k] for k in selected_indices_pos_sl]
        b=[df_neg_ol.Influence.index[k] for k in selected_indices_neg_ol]
        c=[df_neg_sl.Influence.index[k] for k in selected_indices_neg_sl]
        d=[df_pos_ol.Influence.index[k] for k in selected_indices_pos_ol]
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
    simlist=np.array([cosine_similarity(X_test[test_idx].numpy().reshape(1,-1), X_test[i].numpy().reshape(1, -1)).item() for i in range(len(X_test))])
    mostsim=simlist.argsort()[-10:]
    leastsim=simlist.argsort()[:10]
    concatenated_array = np.concatenate((mostsim, leastsim)).tolist()
    cosine_sim=simlist[[concatenated_array]]
    set1=get_explanation(test_idx)
#     mean_ex_similarity=[]
    jaccard_sim=[]
#     mbm_sim=[]
    fuzzy_jac=[]
    for i in tqdm(concatenated_array):
        set2=get_explanation(i)
        jaccard_sim.append(jaccard_similarity(set(set1), set(set2)))
#         similarity_matrix = cosine_similarity(X_train[[set1]], X_train[[set2]])
#         mean_ex_similarity.append(np.mean(similarity_matrix))
        mbm=average_mbm_similarity(X_train[[set1]], X_train[[set2]])
#         mbm_sim.append(mbm)
        fuzzy_jac.append(mbm/(len(set2)/5 - mbm))
          
    return cosine_sim.flatten().tolist(), fuzzy_jac, jaccard_sim


sample_idx = random.sample(range(0, X_test.shape[0]), 100)
cosine_total=[]
# mbm_total=[]
jaccard_total=[]
fuzzy_total=[]
for i in tqdm(sample_idx):
    cosine_sim, fuzzy_jac, jaccard_sim = aide_eval(i)
    cosine_total.append(cosine_sim)
#     mbm_total.append(mbm_sim)
    fuzzy_total.append(fuzzy_jac)
#     mean_total.append(mean_ex_similarity)
    jaccard_total.append(jaccard_sim)
    
# plt.figure()
# plt.scatter(cosine_total, mbm_total, alpha=0.5)
# plt.title("AIDE Faithfulness")
# plt.xlabel('Cosine Similarity of Emails')
# plt.ylabel('Maximum Bipartite Matching Similarity of Explanations')
# plt.legend(fontsize=7)
# plt.savefig('aide_txt_mbm22_vs_cos.eps', format='eps')

np.savez('plotdata/plot_spam.npz', cosine=cosine_total, jaccard=jaccard_total, fuzzy=fuzzy_total)

plt.rcParams.update({'font.size': 16})

plt.figure()
plt.scatter(cosine_total, jaccard_total, alpha=0.5)
plt.xlabel('Cosine Similarity')
plt.ylabel('Jaccard Similarity')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig('aide_txt_jaclast_vs_cos.pdf', bbox_inches='tight')

plt.figure()
plt.scatter(cosine_total, fuzzy_total, alpha=0.5)
plt.xlabel('Cosine Similarity')
plt.ylabel('Fuzzy Jaccard Similarity')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.savefig('aide_txt_fuzjaclast_vs_cos.pdf', bbox_inches='tight')