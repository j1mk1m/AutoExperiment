import json
import torch 
import numpy as np

from evaluation_functions.fairness_metrics import *
from evaluation_functions.calculation_utils import calc_num_rel, calc_E_target, calc_E_system, build_E_collect
from evaluation_functions.load_ranking_models import load_deterministic_ranker

def eval_function_static(save_df, user_label, item_label, matrix_label, args):
    # construct E_target
    num_rel = calc_num_rel(matrix_label)

    # Calculate E_target with user browsing model
    E_target = calc_E_target(args, matrix_label, num_rel)

    # construct E_collect (collectiion of exposures?), E_collect = random exposure
    E_collect = build_E_collect(args, E_target)

    top_item_id = np.array(list(save_df["item"])).reshape(-1, 100)
    
    # put the exposure value into the selected positions
    E_system = calc_E_system(args, E_target, top_item_id)

    E_target = torch.from_numpy(E_target)
    E_collect = torch.from_numpy(E_collect)
    indicator = torch.ones((E_target.shape[0], E_target.shape[1]))
    IIF = II_F(E_system, E_target, E_collect, indicator)
    GIF = GI_F(E_system, E_target, E_collect, user_label, indicator)
    AIF = AI_F(E_system, E_target, E_collect, indicator)
    IGF = IG_F(E_system, E_target, E_collect, item_label, indicator)
    GGF = GG_F(E_system, E_target, E_collect, user_label, item_label, indicator)[:3]
    AGF = AG_F(E_system, E_target, E_collect, item_label, indicator)

    return IIF, GIF, IGF, GGF, AIF, AGF


def compute_static(args, matrix_label, item_mapping, user_label, item_label):
    # Load deterministic ranker
    save_df = load_deterministic_ranker(args, item_mapping)

    save_IIF, save_IGF, save_GIF, save_GGF, save_AIF, save_AGF = [], [], [], [], [], []
    save_IID, save_IGD, save_GID, save_GGD, save_AID, save_AGD = [], [], [], [], [], []
    save_IIR, save_IGR, save_GIR, save_GGR, save_AIR, save_AGR = [], [], [], [], [], []

    IIF_all, GIF_all, IGF_all, GGF_all, AIF_all, AGF_all \
        = eval_function_static(save_df, user_label, item_label, matrix_label, args)

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

    dict = {"IIF": save_IIF, "IGF": save_IGF, "GIF": save_GIF, "GGF": save_GGF, "AIF": save_AIF, "AGF": save_AGF,
            "IID": save_IID, "IGD": save_IGD, "GID": save_GID, "GGD": save_GGD, "AID": save_AID, "AGD": save_AGD,
            "IIR": save_IIR, "IGR": save_IGR, "GIR": save_GIR, "GGR": save_GGR, "AIR": save_AIR, "AGR": save_AGR}

    print(dict)
    for key in dict:
        if args.age == 'Y':
            with open("src/outputs/{}/{}_all_{}_static_Y.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict[key], fp)
        else:
            with open("src/outputs/{}/{}_all_{}_static.json".format(args.data, key, args.model), "w") as fp:
                json.dump(dict[key], fp)