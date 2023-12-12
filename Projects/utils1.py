from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd

def input_data(df, test_idx, Y_test):
    
    scaler = MinMaxScaler()
    
   
    if sum(df.Influence.tolist())==0:
        df_pos = df[df.Similarity>0.3].sort_values('Similarity', ascending=False) #.reset_index(drop=True)
        df_pos_sl = df_pos[df_pos.Y_train==Y_test[test_idx].item()]
    else:
        df_pos = df[df.Influence>0].sort_values('Influence', ascending=False)[:50] #.reset_index(drop=True)
        df_pos= df_pos.sort_values('Similarity', ascending=False)[:30]
        df_pos[['Influence', 'Similarity']]= scaler.fit_transform(df_pos[['Influence', 'Similarity']])
        df_pos_sl = df_pos[df_pos.Y_train==Y_test[test_idx].item()]  #[:n_p]

    return df_pos_sl

    
    
def calculate_cosine_similarity(a, b):
    """Calculate cosine similarity between two arrays."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def greedy_subset_selection(df, N):
    
    arrays = [np.array(i) for i in df.X_train]
    influence_scores = df.Influence.tolist()  # List of influence scores for each array
    prox = df.Similarity.tolist()
#     infsim = df.infsim.tolist()
    n_arrays = len(arrays)
    selected_indices = []
    # Start with the array with the highest influence score


    initial_idx = np.argmax(prox)


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
                    
                    combined_score = 0.3*influence_scores[i]+prox[i] #-0.4*similarity

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

#                     if sett =='positive' and label=='same':
#                         combined_score = 0.3*influence_scores[i]**(1/2) +0.8*prox[i]-0.4*similarity
#                     elif sett =='negative' and label=='opposite':
#                         combined_score = -0.3*influence_scores[i]**(1/2) +0.8*prox[i] -0.4*similarity       
#                     elif sett =='positive' and label=='opposite':
#                         combined_score = 0.3*influence_scores[i]**(1/2)+0.8*prox[i]-0.4*similarity
#                     elif sett =='negative' and label=='same':
#                         combined_score = -0.3*influence_scores[i]**(1/2)+0.8*prox[i]-0.4*similarity

# Interpret 462  max prox pos_ol
# if sett =='positive' and label=='same':
#     combined_score = 0.2*influence_scores[i] +0.8*prox[i]-0.2*similarity
# elif sett =='negative' and label=='opposite':
#     combined_score = -0.1*influence_scores[i] +0.9*prox[i] -0.3*similarity       
# elif sett =='positive' and label=='opposite':
#     combined_score = 0.5*influence_scores[i]+0.4*prox[i]-0.2*similarity
# elif sett =='negative' and label=='same':
#     combined_score = -0.3*influence_scores[i]-0.2*similarity-0.8*prox[i]

# Clarify an ambiguous 458
# if sett =='positive' and label=='same':
#     combined_score = 0.8*influence_scores[i] +0.5*prox[i]-0.2*similarity
# elif sett =='negative' and label=='opposite':
#     combined_score = -0.1*influence_scores[i] +0.9*prox[i] -0.3*similarity       
# elif sett =='positive' and label=='opposite':
#     combined_score = 0.3*influence_scores[i]-0.3*similarity-0.7*prox[i]
# elif sett =='negative' and label=='same':
#     combined_score = -0.3*influence_scores[i]-0.4*similarity-0.7*prox[i]