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
    """ 
    """
    if data == 'ml-1m':
        dic = df.groupby('user')['age'].apply(list).to_dict()

    else:
        dic = df.groupby('user')['nhelpful'].mean().to_dict()

    for id, attribute in dic.items():
        id = int(id)
        if data == 'ml-1m':
            dic[id] = age_mapping_ml1m(attribute[0])
        elif data == 'lt':
            dic[id] = help_mapping(attribute)
        else:
            print('Mapping not avilable for this dataset')

    index_age = [[], [], [], [], [], [], []]
    
    for i in range(0, len(np.max(dic.keys()))):
        try: 
            if 0 == dic[i]:
                    index_age[0].append(i)
            elif 1 == dic[i]:
                index_age[1].append(i)
            elif 2 == dic[i]:
                index_age[2].append(i)
            elif 3 == dic[i]:
                index_age[3].append(i)
            elif 4 == dic[i]:
                index_age[4].append(i)
            elif 5 == dic[i]:
                index_age[5].append(i)
            elif 6 == dic[i]:
                index_age[6].append(i)
        except KeyError:
            pass


    for i in range(len(index_age)):
        index_age[i] = np.array(index_age[i])

    age_type = 7   
    age_mask = torch.zeros(age_type, user_size)
    for i in range(user_size):
        for k in range(age_type):
            if i in index_age[k]:
                age_mask[k][i] = 1


    return index_age, age_mask
