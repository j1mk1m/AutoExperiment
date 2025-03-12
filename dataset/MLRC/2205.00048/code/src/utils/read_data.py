import os
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter
import argparse
from Dataloders.MovieLensLoader import MovieLens1M
from Dataloders.LibraryThingLoader import LibraryThing
from indexation_functions.gender import gender_index
from indexation_functions.age import user_index_7groups
from utils.indexation_functions.occupation import occupation_index
from indexation_functions.librarything import engagement_index
from indexation_functions.genere import genre_ml1m_index
from indexation_functions.popularity import pop_index
import scipy


def preprocessing(args):
    """ Prepare data for processing
    """
    data_dir = os.path.join('src/datasets', args.data)
    
    #following line is commented out because otherwise there is a local var df referenced before assignment error
    if args.data == 'ml-1m':
        df, item_mapping = MovieLens1M(data_dir).load()
    elif args.data == 'lt':
        df, item_mapping = LibraryThing(data_dir, args.ndatapoints).load()
    else:
        print('Please provide valid dataset')

   
    user_size = len(df['user'].unique())
    item_size = len(df['item'].unique())

    """construct matrix of user and group lables
    """
    
    df_rate = df[["user", "item", "rate"]]
    
    # Keep only movies with ratings larger than 3, if the rating is >3 the user would watch it
    df_rate = df_rate[df_rate['rate'] > 3]
    df_rate = df_rate.reset_index().drop(['index'], axis=1)
    
    # Turn to binary value 
    df_rate['rate'] = 1

    # Store as sparse data

    matrix_label = scipy.sparse.csr_matrix(
        (np.array(df_rate['rate']), (np.array(df_rate['user']), np.array(df_rate['item']))))
    
    return df, item_mapping, matrix_label, user_size, item_size







def obtain_group_index(df, args):
    """
    """
    if args.data == 'ml-1m':
        user_size = len(df['user'].unique())
        
        #matrices of where in the df there is an index for each case
        index_F, index_M = gender_index(df)
        #list of size 2 that has the #women and #men
        index_gender = [torch.tensor(index_F).long(), torch.tensor(index_M).long()]
        #an array of arrays for all 7 age groups and an array that has 1 if the user belongs to a specific age group
        index_age, age_mask = user_index_7groups(df, user_size, args.data)
        index_pop, pop_mask = pop_index(df)
        index_occup, occup_mask = occupation_index(df, user_size)
        
        index_genre = []
        index_genre, genre_mask = genre_ml1m_index(df)

        return index_F, index_M, index_gender, index_age, index_genre, index_pop, index_occup, age_mask, pop_mask, occup_mask, genre_mask

    elif args.data == 'lt':
        user_size = len(df['user'].unique())
        #matrices of where in the df there is an index for each group
        index_engagement, engagement_mask = engagement_index(df)
        index_helpful, helpful_mask = user_index_7groups(df, user_size, args.data)

        return index_engagement, index_helpful, engagement_mask, helpful_mask



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="lt")
    parser.add_argument("--ndatapoints", type=int, default=5000)
    return parser.parse_args()


if __name__ == '__main__':

    args = parser_args()
    df, item_mapping, matrix_label, user_size, item_size = preprocessing(args)
    
    if args.data == 'ml-1m':
        index_F, index_M, index_gender, index_age, index_genre, index_pop, index_occup, age_mask, pop_mask, occup_mask, genre_mask  = obtain_group_index(df, args)
    elif args.data == 'lt':
        index_engagement, index_helpful, engagement_mask, helpful_mask = obtain_group_index(df, args)
