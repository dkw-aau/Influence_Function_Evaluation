from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd

def input_data(df, test_idx, Y_test, sett=None):
    
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
    assert(sett!=None, 'Input the sett')
    
    if sett == 'positive':

        df_pos = df[df.Influence>0].sort_values('Influence', ascending=False) #.reset_index(drop=True)
        df_pos[['Influence', 'Similarity']]= scaler.fit_transform(df_pos[['Influence', 'Similarity']])
        df_pos_sl = df_pos[df_pos.Y_train==Y_test[test_idx].item()][:n_p]
        
        df_pos_ol = df_pos[df_pos.Y_train!=Y_test[test_idx].item()][:n_p]
        df_pos_ol['infsim']=df_pos_ol['Influence']+df_pos_ol['Similarity']
        df_pos_sl['infsim']=df_pos_sl['Influence']+df_pos_sl['Similarity']
        return df_pos_sl, df_pos_ol
    
    elif sett=='negative':
       
        df_neg = df[df.Influence<0].sort_values('Influence', ascending=True) #.reset_index(drop=True)
        df_neg[['Influence', 'Similarity']]= scaler.fit_transform(df_neg[['Influence', 'Similarity']])
        df_neg_ol = df_neg[df_neg.Y_train!=Y_test[test_idx].item()][:nn]
        df_neg_sl = df_neg[df_neg.Y_train==Y_test[test_idx].item()][:nn]
        df_neg_ol['infsim']=df_neg_ol['Influence']+df_neg_ol['Similarity']
        df_neg_sl['infsim']=df_neg_sl['Influence']+df_neg_sl['Similarity']
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
    infsim = df.infsim.tolist()
    n_arrays = len(arrays)
    selected_indices = []
    # Start with the array with the highest influence score
    if sett =='positive' and label=='opposite':
        initial_idx = np.argmax(prox)
    elif sett =='positive' and label=='same':
        initial_idx = np.argmax(prox)
    elif sett =='negative' and label=='opposite':
        initial_idx = np.argmax(prox)
    elif sett =='negative' and label=='same':
        initial_idx = np.argmin(prox)
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
                if any(i>0.85 for i in final_list):
                    continue
                else:
                    
                    similarity = np.mean(final_list)
                    # Calculate combined score of diversity and influence score
                    if sett =='positive' and label=='same':
                        combined_score = 0.2*influence_scores[i] +0.8*prox[i]-0.2*similarity
                    elif sett =='negative' and label=='opposite':
                        combined_score = -0.1*influence_scores[i] +0.9*prox[i] -0.3*similarity       
                    elif sett =='positive' and label=='opposite':
                        combined_score = 0.7*influence_scores[i]+0.5*prox[i]
                    elif sett =='negative' and label=='same':
                        combined_score = -0.3*influence_scores[i]-0.2*similarity-0.8*prox[i]
                        
                    # Update selected array if it provides the highest gain
                    if combined_score > max_gain:
                        max_gain = combined_score
                        selected_idx = i
        
        # Add selected array to the subset
        selected_indices.append(selected_idx)
    
    return selected_indices



# Wrong 300
# if sett =='positive' and label=='same':
#     combined_score = 0.2*influence_scores[i] +0.8*prox[i]-0.2*similarity
# elif sett =='negative' and label=='opposite':
#     combined_score = -0.1*influence_scores[i] +0.9*prox[i] -0.3*similarity       
# elif sett =='positive' and label=='opposite':
#     combined_score = 0.3*influence_scores[i]-0.3*similarity-0.8*prox[i]
# elif sett =='negative' and label=='same':
#     combined_score = -0.3*influence_scores[i]-0.2*similarity-0.8*prox[i]