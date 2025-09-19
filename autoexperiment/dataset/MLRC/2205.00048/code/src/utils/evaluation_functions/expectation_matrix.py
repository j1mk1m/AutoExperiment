import json
import torch 
import numpy as np
from tqdm import trange
from scipy.special import softmax

from evaluation_functions.fairness_metrics import *
from evaluation_functions.calculation_utils import calc_num_rel, calc_E_target, normalize_matrix_by_row
from evaluation_functions.load_ranking_models import load_deterministic_ranker

def compute_exp_matrix(args, matrix_label, item_mapping, user_label, item_label):
    save_df = load_deterministic_ranker(args, item_mapping)

    if args.model == 'LDA':
        save_df["score"] = save_df["score"] * 1000
    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    # rand_tau_list = [2, 4, 8, 16]
    rand_tau_list = [0.125, 8]
    len_tau = len(rand_tau_list)

    """evaluate on whole"""
    for i in range(len_tau):
        rand_tau = rand_tau_list[i]
        print("tau={}".format(rand_tau))
        # construct E_target
        num_rel = calc_num_rel(matrix_label)

        # Calculate E_target with user browsing model
        E_target = calc_E_target(args, matrix_label, num_rel)

        # construct E_collect
        E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()


        top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
        top_score = np.array(list(save_df["score"])).reshape(-1, 100)
        # This was commented out at some point
        top_score = normalize_matrix_by_row(top_score)
        weight = softmax(top_score / rand_tau, axis=1)

        # put the exposure value into the selected positions
        sample_times = 100
        E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
        for _ in trange(sample_times, ascii=False):
            E_system_tmp = np.zeros((E_target.shape[0], E_target.shape[1]))
            exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
            for i in range(len(top_item_id)):
                tmp_selected = np.random.choice(top_item_id[i], 100, replace=False, p=weight[i])
                tmp_selected = np.array([int(j) for j in tmp_selected])
                E_system_tmp[i][tmp_selected] = exp_vector
            E_system += E_system_tmp
        E_system /= sample_times

        E_system = torch.from_numpy(E_system)
        E_target = torch.from_numpy(E_target)
        E_collect = torch.from_numpy(E_collect)
        indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
        GG_target_stochas = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[3]
        GG_system_stochas = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[4]

        with open("src/outputs/{}/GG_MT_{}_{}.json".format(args.data, rand_tau, args.model), "w") as fp:
            json.dump(np.array(GG_target_stochas).tolist(), fp)
        with open("src/outputs/{}/GG_MS_{}_{}.json".format(args.data, rand_tau, args.model), "w") as fp:
            json.dump(np.array(GG_system_stochas).tolist(), fp)

    # construct E_target
    num_rel = calc_num_rel(matrix_label)

    exposure_rel = (args.gamma / (1.0 - args.gamma)) * (1.0 - np.power(args.gamma, num_rel).astype("float"))
    E_target = exposure_rel / num_rel * matrix_label

    # construct E_collect
    E_collect = np.ones((E_target.shape[0], E_target.shape[1])) * E_target.mean()

    # construct E_system
    user_size = E_target.shape[0]

    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    top_score = np.array(list(save_df["score"])).reshape(-1, 100)

    # put the exposure value into the selected positions
    E_system = np.zeros((E_target.shape[0], E_target.shape[1]))
    exp_vector = np.power(args.gamma, np.arange(100) + 1).astype("float")
    for i in range(len(top_item_id)):
        top_item_id = [list(map(int, i)) for i in top_item_id]
        E_system[i][top_item_id[i]] = exp_vector

    E_system = torch.from_numpy(E_system)
    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
    GG_target_static = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[3]
    GG_system_static = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[4]
    GG_collect = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[5]

    print("GG_target_static:", GG_target_static)
    print("GG_system_static:", GG_system_static)
    print("GG_collect:", GG_collect)

    with open("src/outputs/{}/GG_MT_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_target_static).tolist(), fp)
    with open("src/outputs/{}/GG_MS_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_system_static).tolist(), fp)
    with open("src/outputs/{}/GG_collect_{}_static.json".format(args.data, args.model), "w") as fp:
        json.dump(np.array(GG_collect).tolist(), fp)

