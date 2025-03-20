import torch 
import numpy as np
from utils.indexation_functions.librarything import help_mapping

def age_mapping_ml1m(age):
    if age == 1:
        return 0
    elif age == 18:
        return 1
    elif age == 25:
        return 2
    elif age == 35:
        return 3
    elif age == 45:
        return 4
    elif age == 50:
        return 5
    elif age == 56:
        return 6
    else:
        print('Error in age data, age set = ', age)


def user_index_7groups(df, user_size, data):
    if data == 'ml-1m':
        dic = df.groupby('user')['age'].apply(list).to_dict()
        
        # Debugging Output
        print("Dictionary content:", dic)  # Check the contents
        print("Length of Dictionary:", len(dic))  # Check the number of user entries
        print("User Size:", user_size)  
        
        age_indices = [dic.get(id, -1) for id in range(user_size)]  # Use get() to avoid KeyError, fill with -1
        print("Age Indices Count:", len(age_indices))  # Verify final length

        # Print the first few age indices to verify the output
        print("First few Age Indices:", age_indices[:10])

        return torch.tensor(age_indices, dtype=torch.long)  # Ensure returned value is a tensor
    else:
        dic = df.groupby('user')['nhelpful'].mean().to_dict()

    for id, attribute in dic.items():
        id = int(id)
        if data == 'ml-1m':
            dic[id] = age_mapping_ml1m(attribute[0])
        elif data == 'lt':
            dic[id] = help_mapping(attribute)
        else:
            print('Mapping not available for this dataset')
