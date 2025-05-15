import torch
import datasets
import torchvision
import matplotlib.pyplot as plt
import bcos
import bcos.modules
import bcos.data
import bcos.data.transforms
import attribution_methods
import utils
import model_activators
import numpy as np
import pandas as pd
import seaborn as sns
import hubconf
from PIL import Image
from captum.attr import visualization as viz
from visualization_function import plot_contribution_map
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

def contribution_map_plot(checkpoint="qualitative_results/model_checkpoint_pareto_0.6883_0.4277_0.0126_10.pt"):
    is_bcos = True
    is_vanilla = False
    val_batch_size = 4
    layer_idx = None
    
    if is_bcos:
        transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    val_data_processed = datasets.VOCDetectParsed(
        root='datasets/COCO2014/processed', image_set="val", transform=transformer)

    processed_val_loader = torch.utils.data.DataLoader(
        val_data_processed, batch_size=val_batch_size, shuffle=False, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)
    
    batch, labels, bboxes = next(iter(processed_val_loader))
    batch.requires_grad = True
    gt_classes = utils.get_random_optimization_targets(labels)
    
    if is_bcos:
        model = hubconf.resnet50(pretrained=False)
        model[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
                        in_channels=model[0].fc.in_channels, out_channels=80)
        layer_dict = {"Input": None, "Mid1": 3, "Mid2": 4, "Mid3": 5, "Final": 6}
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
    elif is_vanilla:
        model = torchvision.models.resnet50(
                    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(
                in_features=model.fc.in_features, out_features=80)
        layer_dict = {"Input": None, "Mid1": 4, "Mid2": 5, "Mid3": 6, "Final": 7}
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))

    model.load_state_dict(state_dict['model'])
    model_activator = model_activators.ResNetModelActivator(
            model=model, layer=layer_idx, is_bcos=is_bcos)

    eval_attributor = attribution_methods.get_attributor(
                model, 'BCos', False, False, False, (224, 224), batch_mode=True)
    bb_list = utils.filter_bbs(bboxes[3], gt_classes[3])
    logits, features = model_activator(batch)
    attributions = eval_attributor(features, logits, classes=gt_classes).squeeze(1)

    # Display first image
    ax, im = plot_contribution_map(attributions[3].clamp(min=0),
                                   plot_bbs=True,
                                   bb_list=bb_list)
    plt.show()

    # Display second image with blended heat map
    ax, im = plot_contribution_map(attributions[3].clamp(min=0),
                                   blended_heat_map=True,
                                   original_image=batch[3, 3:, ...].detach().numpy().transpose(1,2,0),
                                   plot_bbs=True,
                                   bb_list=bb_list)
    plt.show()


def plot_epg_vs_bbsize(data):
    loaded_df = pd.read_csv(data)
    epg_scores = np.array(loaded_df['epg_scores'])
    bbsizes = np.array(loaded_df['bb_sizes'])
    correlation_coefficient, p_value = pearsonr(bbsizes, epg_scores)
    print(f"Correlation Coefficient: {correlation_coefficient}")
    print(f"P-value: {p_value}")


    num_bins = 10
    hist, bin_edges = np.histogram(bbsizes, bins=num_bins)

    epg_values = [[] for _ in range(num_bins)]

    for i in range(num_bins):
        bin_mask = (bbsizes >= bin_edges[i]) & (bbsizes < bin_edges[i + 1])
        epg_values[i] = epg_scores[bin_mask]

    sns.set(style="whitegrid", font_scale=1.5)
    plt.figure(figsize=(12, 8))  

    ax = sns.boxplot(data=epg_values, color=sns.color_palette("deep")[0], showmeans=True, meanprops={"marker":"o",
                        "markerfacecolor":"white", 
                        "markeredgecolor":"black"},
                        showfliers=False)

    # Customize x-axis tick labels
    x_tick_labels = ['0 - 5k', '5k - 10k', '10k - 15k', '15k - 20k', '20k - 25k', '25k - 30k', '30k - 55k', '  35k - 40k', '  40k - 45k', '  45k - 50k']
    ax.set_xticklabels(x_tick_labels, rotation=60, ha='center')

    plt.xlabel('Total Bounding Box Size in Pixels', fontsize=18)
    plt.ylabel('EPG Score', fontsize=18)
    plt.tight_layout()

    #plt.savefig('corr_boxplot_FINAL.png', dpi=300)
    plt.show()