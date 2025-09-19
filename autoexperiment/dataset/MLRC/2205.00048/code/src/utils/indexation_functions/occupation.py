import torch 
import numpy as np 

def occupation_index(df, user_size):
    occup_dic = df.groupby('user')['occupation'].apply(list).to_dict()
    occupations = df['occupation'].unique().astype(int) # List the available occupations
    index_occup = {key: [] for key in occupations}

    for i in range(0, len(occup_dic)):
        occupation = int(occup_dic[i][0])
        index_occup[occupation].append(i)

    for i in range(len(index_occup)):
        index_occup[i] = np.array(index_occup[i])
    
    occup_mask = torch.zeros(len(occupations), user_size)
    for i in range(user_size):
        for k in range(len(occupations)):
            if i in index_occup[k]:
                occup_mask[k][i] = 1
    
    return index_occup, occup_mask


