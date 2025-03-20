
import torch 
import numpy as np 

def genre_ml1m_index(df):
    df_genre = df[['item', 'genre']]
    df_genre = df_genre.drop_duplicates(subset=['item'], keep='first').reset_index(drop=True)
    ls = df_genre['genre'].tolist()
    for i in range(len(ls)):
        ls[i] = ls[i].split("|")
    
    for i in range(len(ls)):
        for j in range(len(ls[i])):
            if ls[i][j] == 'Action':
                ls[i][j] = 0
            elif ls[i][j] == 'Adventure':
                ls[i][j] = 1
            elif ls[i][j] == 'Animation':
                ls[i][j] = 2
            elif ls[i][j] == "Children's":
                ls[i][j] = 3
            elif ls[i][j] == 'Comedy':
                ls[i][j] = 4
            elif ls[i][j] == 'Crime':
                ls[i][j] = 5
            elif ls[i][j] == 'Documentary':
                ls[i][j] = 6
            elif ls[i][j] == 'Drama':
                ls[i][j] = 7
            elif ls[i][j] == 'Fantasy':
                ls[i][j] = 8
            elif ls[i][j] == 'Film-Noir':
                ls[i][j] = 9
            elif ls[i][j] == 'Horror':
                ls[i][j] = 10
            elif ls[i][j] == 'Musical':
                ls[i][j] = 11
            elif ls[i][j] == 'Mystery':
                ls[i][j] = 12
            elif ls[i][j] == 'Romance':
                ls[i][j] = 13
            elif ls[i][j] == 'Sci-Fi':
                ls[i][j] = 14
            elif ls[i][j] == 'Thriller':
                ls[i][j] = 15
            elif ls[i][j] == 'War':
                ls[i][j] = 16
            elif ls[i][j] == 'Western':
                ls[i][j] = 17


    genre_mask = torch.zeros(18, len(df_genre)) # [18, 3706]
    for i in range(len(df_genre)):
        for k in range(0, 18):
            if k in ls[i]:
                genre_mask[k][i] = 1

    index_genre = [] #list 18
    for i in range(genre_mask.shape[0]):
        index_genre.append(torch.tensor(np.where(genre_mask[i] == 1)[0]).long())
    
    return index_genre, genre_mask
