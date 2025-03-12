import time
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from utils.parser import parser_args
from utils.evaluation_functions.fairness_metrics import *
from utils.read_data import preprocessing, obtain_group_index
from utils.evaluation_functions.stochastic import compute_stochas
from utils.evaluation_functions.static import compute_static
from utils.evaluation_functions.expectation_matrix import compute_exp_matrix

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


if __name__ == '__main__':
    print('start timer')
    start = time.time()

    args = parser_args()
    args.device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", args.device)

    """read data and assign attribute index"""
    df, item_mapping, matrix_label, user_size, item_size = preprocessing(args)
    
    # Obtain group indices for Movielens dataset
    if args.data == 'ml-1m':
        index_F, index_M, index_gender, index_age, index_genre, index_pop, index_occup, age_matrix, pop_mask, occup_matrix, genre_matrix \
            = obtain_group_index(df, args)
        
        # Set user group lable:
        if args.age == 'Y':
            user_label = age_matrix #[7,6040]
        elif args.age == 'Occup':
            user_label = occup_matrix
        else:
            # Build matrix with gender information
            gender_matrix = torch.zeros(2, len(index_F) + len(index_M)) #[2, #females + #males] , 1st row for F 2nd for M
            for ind in index_F:
                gender_matrix[0][ind] = 1
            for ind in index_M:
                gender_matrix[1][ind] = 1
    
            user_label = gender_matrix  # .to(args.device) [2,6040]

        # Set item group lable
        item_label = genre_matrix  # .to(args.device) [18, 3706]

    # Obtain group indices for LibraryThing dataset
    elif (args.data == 'lt'):
        index_engagement, index_helpful, engagement_matrix, helpful_matrix = obtain_group_index(df, args) 
        user_label = helpful_matrix 
        item_label = engagement_matrix

    print(matrix_label.shape[0])
    matrix_label = np.array(matrix_label.todense()) #rating matrix for matrix factorization, user-item relevance matrix Y [6040, 3706]
    print('mat lab', matrix_label.shape)
    # Print set-up statics
    print('-------- Configuration of Experiment --------')
    print("norm:", args.norm)
    print("coll:", args.coll)
    print("model:", args.model)
    print('conduct:', args.conduct)
    print('----------------------------------------------')

    # Run computation according to conduct nide
    if args.conduct == 'sh':
        compute_stochas(args, matrix_label, item_mapping, user_label, item_label)
    elif args.conduct == 'st':
        compute_static(args, matrix_label, item_mapping, user_label, item_label)

    # Compute Expectation matrix
    #compute_exp_matrix(args, matrix_label, item_mapping, user_label, item_label)

    stop = time.time()
    print('Time elapsed: ', np.round(stop-start, 4), ' s')
    
    
    # Write experiment GPU time to file
    with open("src/outputs/Experiment_times.txt", "a") as f:
        f.write(args.model + " " + args.age + " " + args.conduct + " " + str(np.round(stop-start, 4)) + " s  \n")
    
    print('---------------------------------------------')
    print("Experiment completed successfully")
    print('---------------------------------------------')