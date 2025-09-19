import numpy as np 

def gender_index(df):
    """  Turn gender lable into index system
    return:
        index_F: List of index of Females
        index_M: List of index of Males
    """

    gender_dic = df.groupby('user')['gender'].apply(list).to_dict()
    index_F, index_M = [], []

    for i in range(0, len(gender_dic)):
        if 'f' in gender_dic[i] or 'F' in gender_dic[i]:
            index_F.append(i)
        else:
            index_M.append(i)

    return np.array(index_F), np.array(index_M)
