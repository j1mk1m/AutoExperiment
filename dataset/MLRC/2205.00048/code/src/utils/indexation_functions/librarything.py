import torch 
import numpy as np

def help_mapping(nhelpful):
    if 0 <= nhelpful < 1 :
        return 0
    elif 1 <= nhelpful < 2:
        return 1
    elif 2 <= nhelpful < 3:
        return 2
    elif 3 <= nhelpful < 4:
        return 3
    elif 4 <= nhelpful < 5:
        return 4
    elif 5 <= nhelpful < 6:
        return 5
    elif nhelpful >= 6:
        return 6
    else:
        print('Error in nhelpful data, nhelpful set = ', nhelpful)

def engagement_index(df):
    df_length = df[['item', 'commentlength']]
    print(len(df_length))
    df_length = df_length.groupby('item')['commentlength'].mean().reset_index()
    ls = df_length['commentlength'].tolist()

    for i in range(len(ls)):
        if ls[i] <= 50:
            ls[i] = 0
        elif 50 < ls[i] <= 100:
            ls[i] = 1
        elif 100 < ls[i] <= 150:
            ls[i]= 2
        elif 150 < ls[i] <= 200:
            ls[i] = 3
        elif 200 < ls[i] <= 250:
            ls[i] = 4
        elif 250 < ls[i] <= 300:
            ls[i] = 5
        elif 300 < ls[i] <= 350:
            ls[i] = 6
        elif 350 < ls[i] <= 400:
            ls[i] = 7
        elif 400 < ls[i] <= 450:
            ls[i] = 8
        elif 450 < ls[i] <= 500:
            ls[i] = 9
        elif 500 < ls[i] <= 550:
            ls[i] = 10
        elif 550 < ls[i] <= 600:
            ls[i] = 11
        elif 600 < ls[i] <= 650:
            ls[i] = 12
        elif 650 < ls[i] <= 700:
            ls[i] = 13
        elif 700 < ls[i] <= 750:
            ls[i] = 14
        elif 750 < ls[i] <= 800:
            ls[i] = 15
        elif 800 < ls[i] <= 850:
            ls[i] = 16
        elif ls[i] > 850:
            ls[i] = 17

    genre_mask = torch.zeros(18, len(df_length)) 
    for i in range(len(df_length)):
        for k in range(0, 18):
            if k in ls:
                genre_mask[k] = 1

    index_genre = [] #list 18
    for i in range(genre_mask.shape[0]):
        index_genre.append(torch.tensor(np.where(genre_mask[i] == 1)[0]).long())
    print('engagement mask', genre_mask.shape)
    return index_genre, genre_mask

