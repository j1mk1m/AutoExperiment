import json
import torch 
import numpy as np
from scipy.special import softmax

from evaluation_functions.fairness_metrics import *
from evaluation_functions.calculation_utils import calc_num_rel, calc_E_target, calc_E_system, build_E_collect, normalize_matrix_by_row
from evaluation_functions.load_ranking_models import load_deterministic_ranker

def eval_function_stochas(save_df, user_label, item_label, matrix_label, args, rand_tau=1):
    # construct E_target
    num_rel = calc_num_rel(matrix_label)
    
    # user browsing model  
    E_target = calc_E_target(args, matrix_label, num_rel)

    # construct E_collect
    E_collect = build_E_collect(args, E_target)

    # To pytorch tensors 
    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)

    print(len(save_df['item']))
    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100) #[6040, 100]
    top_score = np.array(list(save_df["score"])).reshape(-1, 100)
    print('top_item_id', top_item_id.shape)
    print('top score ', top_score.shape)
    if args.norm == 'Y':
        top_score = normalize_matrix_by_row(top_score)
    weight = softmax(top_score / rand_tau, axis=1) #Y/b in quation of p(d|u)
    
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))

    E_system = calc_E_system(args, E_target, top_item_id, weight=weight)

    IIF_all = II_F(E_system, E_target, E_collect, indicator)
    GIF_all = GI_F(E_system, E_target, E_collect, user_label, indicator)
    AIF_all = AI_F(E_system, E_target, E_collect, indicator)
    IGF_all = IG_F(E_system, E_target, E_collect, item_label, indicator)
    GGF_all = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[:3]
    AGF_all = AG_F(E_system, E_target, E_collect, item_label, indicator)
    print('Metric evaluation complete')
    return IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all  # , IIF_sp, IGF_sp, GIF_sp, GGF_sp, AIF_sp, AGF_sp
    



def compute_stochas(args, matrix_label, item_mapping, user_label, item_label):
    save_df = load_deterministic_ranker(args, item_mapping)
    if args.model == 'LDA':
        save_df["score"] = save_df["score"] * 1000
    elif args.model == 'Pop':
        save_df["score"] = save_df["score"] * 10
    elif args.model in ["PLSA", "RM1", "RSV", "CHI2", "HT", "KLD", "SVD", "UIR", "RM2", "LMWU", "LMWI", "NNU", "NNI"]:
        args.norm = 'Y'
    
    # List of beta_values
    # rand_tau_list = [8, 4, 2, 1, 0.5, 0.25, 0.125] # different values for beta
    rand_tau_list = [8] # different values for beta
    
    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    len_tau = len(rand_tau_list)

    """evaluate on whole"""
    for epoch in range(args.r_ep):
        print("epoch:", epoch)
        for i in range(len_tau):
            print("tau={}".format(rand_tau_list[i]))

            # IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all, IIF_sp, IGF_sp, GIF_sp, GGF_sp, AIF_sp, AGF_sp \
            IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all \
                = eval_function_stochas(save_df, user_label, item_label, matrix_label, args, rand_tau=rand_tau_list[i])

            save_IIF.append(IIF_all[0].item())
            save_GIF.append(GIF_all[0].item())
            save_IGF.append(IGF_all[0].item())
            save_GGF.append(GGF_all[0].item())
            save_AIF.append(AIF_all[0].item())
            save_AGF.append(AGF_all[0].item())
 
            save_IID.append(IIF_all[1].item())
            save_GID.append(GIF_all[1].item())
            save_IGD.append(IGF_all[1].item())
            save_GGD.append(GGF_all[1].item())
            save_AID.append(AIF_all[1].item())
            save_AGD.append(AGF_all[1].item())

            save_IIR.append(IIF_all[2].item())
            save_GIR.append(GIF_all[2].item())
            save_IGR.append(IGF_all[2].item())
            save_GGR.append(GGF_all[2].item())
            save_AIR.append(AIF_all[2].item())
            save_AGR.append(AGF_all[2].item())

    dict_all = {"IIF": save_IIF, "IGF": save_IGF, "GIF": save_GIF, "GGF": save_GGF, "AIF": save_AIF, "AGF": save_AGF,
                "IID": save_IID, "IGD": save_IGD, "GID": save_GID, "GGD": save_GGD, "AID": save_AID, "AGD": save_AGD,
                "IIR": save_IIR, "IGR": save_IGR, "GIR": save_GIR, "GGR": save_GGR, "AIR": save_AIR, "AGR": save_AGR}

    print(dict_all)
    # Save files in json format
    for key in dict_all:
        if args.age == 'Y':
            with open("src/outputs/{}/{}_all_{}_Y.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict_all[key], fp)
        else:
            with open(
                    "src/outputs/{}/{}_all_{}.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict_all[key], fp)
    

    return dict_all