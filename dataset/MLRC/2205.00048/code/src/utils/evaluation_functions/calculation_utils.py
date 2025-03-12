import numpy as np 
import torch
from tqdm import trange 

def normalize_matrix_by_row(matrix):
    """ Normalize matrix per row
    Input: 
        matrix: matrix to be normalized
    Output:
        normalized_matrix: input matrix normalized by row
    """
    sum_of_rows = matrix.sum(axis=1)
    normalized_matrix = matrix / sum_of_rows[:, np.newaxis]
    return normalized_matrix

def calc_num_rel(matrix_label):
    """ Calculate the exponent of 
    Input:
        matrix_label: matrix of labels

    Output:
        num_rel: Exponent to patients factor in the RBP model
    """

    num_rel = matrix_label.sum(1, keepdims=True).reshape(-1, 1).astype("float")#Y sum in 1st dim of rating matrix 6040x1
    num_rel[num_rel == 0.0] = 1.0
    
    return num_rel


def calc_E_target(args, matrix_label, num_rel):
    """ Calculate E_target with user browsing model (USM)
    """
    usm_exposure = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float")) #6040x1
    E_target = usm_exposure / num_rel * matrix_label #[6040,3706]
    return E_target

def build_E_collect(args, E_target):
    """ Calculate E_collect
    """
    if args.coll == 'Y':
        E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean() #[6040,3706]
    else:
        E_collect = np.zeros((E_target.shape[0], E_target.shape[1]))
    return E_collect

def calc_E_system(args, E_target, top_item_id, weight = np.nan):
    """ Calculate E_system
    """

    E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    
    if args.conduct == 'st':
        exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
        for i in range(len(top_item_id)):
            top_item_id = [list(map(int, i)) for i in top_item_id]
            E_system[i][top_item_id[i]] = exp_vector
        
        return  torch.from_numpy(E_system)
    
    if args.conduct == 'sh':
        sample_times = args.s_ep
        for sample_epoch in trange(sample_times, ascii=False): # sample 100 rankings for each user 
            E_system_tmp = np.zeros((E_target.shape[0], E_target.shape[1]))
            exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")  # pre-compute the exposure_vector (100x1)
            for i in range(len(top_item_id)):
                tmp_selected = np.random.choice(top_item_id[i], 100, replace=False, p=weight[i]) #selects one permutation of 100 movies from /
                #top 100 movies from a user's rank with probability weights[user] (100x1)
                
                tmp_selected = np.array([int(j) for j in tmp_selected])
                E_system_tmp[i][tmp_selected] = exp_vector
            E_system += E_system_tmp
        E_system /= sample_times

        return torch.from_numpy(E_system)