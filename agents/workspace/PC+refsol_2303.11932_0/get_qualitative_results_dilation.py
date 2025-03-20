import torch
import datasets
import torchvision
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import bcos
import bcos.modules
import bcos.data
import bcos.data.transforms
import attribution_methods
import utils
import model_activators
import matplotlib.cm as cm
import numpy as np
import hubconf
from captum.attr import visualization as viz
import fixup_resnet
import losses
import argparse
from tqdm import tqdm
from visualization_function import plot_contribution_map
import json

def get_model(is_bcos, is_xdnn, is_vanilla, num_classes):
    if is_bcos:
        model = hubconf.resnet50(pretrained=True)
        model[0].fc = bcos.modules.bcosconv2d.BcosConv2d(
                    in_channels=model[0].fc.in_channels, out_channels=num_classes)
        layer_dict = {"Input": None, "Mid1": 3,
                        "Mid2": 4, "Mid3": 5, "Final": 6}
    elif is_xdnn:
        model = fixup_resnet.xfixup_resnet50()
        imagenet_checkpoint = torch.load(os.path.join("weights/xdnn/xfixup_resnet50_model_best.pth.tar"))
        imagenet_state_dict = utils.remove_module(
            imagenet_checkpoint["state_dict"])
        model.load_state_dict(imagenet_state_dict)
        model.fc = torch.nn.Linear(
            in_features=model.fc.in_features, out_features=num_classes)
        layer_dict = {"Input": None, "Mid1": 3,
                        "Mid2": 4, "Mid3": 5, "Final": 6}
    elif is_vanilla:
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Linear(
                in_features=model.fc.in_features, out_features=num_classes)
        layer_dict = {"Input": None, "Mid1": 4,
                        "Mid2": 5, "Mid3": 6, "Final": 7}
    else:
        raise NotImplementedError
    return model, layer_dict

def main(args):
    seed = 42

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_classes_dict = {"VOC2007": 20, "COCO2014":  80}
    num_classes = num_classes_dict[args.dataset]
        
    is_bcos = (args.model_backbone == "bcos")
    is_xdnn = (args.model_backbone == "xdnn")
    is_vanilla = (args.model_backbone == "vanilla")

    model_dict = {}

    if args.comparison_criterion == 'loss_functions':
        criterion = args.localization_loss_fn
    elif args.comparison_criterion == 'dilations':
        criterion = args.dilations

    for layer, checkpoint, crit_value in zip(args.layer, args.checkpoints_path, criterion):
        if args.comparison_criterion == 'loss_functions':
            if crit_value == 'Baseline':
                loss_loc = losses.get_localization_loss('L1')
            else:
                loss_loc = losses.get_localization_loss(crit_value)
        else:
            loss_loc = losses.get_localization_loss(
                args.localization_loss_fn) if args.localization_loss_fn else None
            
        model, layer_dict = get_model(is_bcos, is_xdnn, is_vanilla, num_classes)
        layer_idx = layer_dict[layer]
        checkpoint = torch.load(f'{checkpoint}')
        model.load_state_dict(checkpoint['model'])
        #model = model.cuda()
        model.eval()
        if layer in model_dict.keys():
            # store model in model_dict
            model_dict[layer][crit_value] = {'model': model}
        else:
            # create new subdict for layer, store model
            model_dict[layer] = {crit_value: {'model': model}}

        # get model_activator and store it in model_dict
        model_activator = model_activators.ResNetModelActivator(model=model, layer=layer_idx, is_bcos=is_bcos)
        model_dict[layer][crit_value]['model_activator'] = model_activator

        # get eval_attributor and store it in model_dict
        interpolate = True if layer_idx is not None else False
        eval_attributor = attribution_methods.get_attributor(
                model, args.attribution_method, loss_loc.only_positive, loss_loc.binarize, interpolate, (224, 224), batch_mode=False)
        model_dict[layer][crit_value]['eval_attributor'] = eval_attributor
        
    
    # get transformation
    if is_bcos:
            transformer = bcos.data.transforms.AddInverse(dim=0)
    else:
        transformer = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        
    root = os.path.join(args.data_path, args.dataset, "processed")

    test_data = datasets.VOCDetectParsed(
        root=root, image_set="test", transform=transformer)
    
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.eval_batch_size, shuffle=False, num_workers=4, collate_fn=datasets.VOCDetectParsed.collate_fn)
    
    #loss_loc = losses.get_localization_loss(
    #    args.localization_loss_fn) if args.localization_loss_fn else None
        
    img_counter = 0
    for batch_idx, (test_X, test_y, test_bbs) in enumerate(tqdm(test_loader)):
        test_X.requires_grad = True
        #test_X = test_X.cuda()
        #test_y = test_y.cuda()
        if torch.sum(test_y) == 0.:
            continue
        gt_classes = utils.get_random_optimization_targets(test_y)  # get one class per image

        for img_idx in range(len(test_X)):
            # check whether image contains bounding box information
            if test_bbs[img_idx] is None:
                continue
            # get corresponding bounding boxes for one class
            bb_list = utils.filter_bbs(
                test_bbs[img_idx], gt_classes[img_idx])

            img_path = f'qualitative_results/{args.comparison_criterion}/{args.model_backbone}/{gt_classes[img_idx]}'
            # create class directory if necessary
            os.makedirs(img_path, exist_ok=True)
            num_class_examples = len(os.listdir(img_path))
            # create subdirectory with index in class (num_subdirectories) and total image number of all testset images (img_counter)
                # (may not be accurate; if image does not contain bounding box, it is skipped)
            img_path += f'/img_num_{num_class_examples}_total_num_{img_counter}'
            img_counter += 1
            os.makedirs(img_path)

            # get original image by inverting transformation
            if is_bcos:
                orig_image = test_X[img_idx, :3, :, :]
            else:
                m, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                #orig_image = (test_X[img_idx] * torch.tensor(std).view(3, 1, 1).cuda()) + torch.tensor(m).view(3, 1, 1).cuda()
                orig_image = (test_X[img_idx] * torch.tensor(std).view(3, 1, 1)) + torch.tensor(m).view(3, 1, 1)
            # save original image as orig_img.png
            print(f"Saving original image at: {os.path.join(img_path, 'orig_img.png')}")
            #plt.imsave(f'{img_path}/orig_img.png', orig_image.cpu().detach().numpy().transpose(1,2,0))
            plt.imsave(f'{img_path}/orig_img.png', orig_image.detach().numpy().transpose(1,2,0))
            _, ax = plt.subplots(1)
            ax.imshow(orig_image.detach().numpy().transpose(1,2,0))
            for bbox in bb_list:
                xmin, ymin, xmax, ymax = bbox
                width = xmax - xmin
                height = ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='b', facecolor='none')
                ax.add_patch(rect)
            ax.axis('off')
            plt.savefig(f'{img_path}/orig_img_with_bbs.png', bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            # save bounding box coordinates
            with open(f'{img_path}/bbox_coords.json', 'w') as f:
                json.dump(bb_list, f)

            # go through different models and save attributions
            for layer in model_dict.keys():
                print(f'Now evaluating injections at layer: {layer}')
                os.makedirs(f'{img_path}/{layer}')
                for crit_value in model_dict[layer]:
                    print(f'Now evaluating injections with criterion: {crit_value}')
                    # get model activator and use it to get logits and features
                    model_activator = model_dict[layer][crit_value]['model_activator']
                    logits, features = model_activator(test_X[img_idx].unsqueeze(0))
                    # get eval attributor 
                    attributor = model_dict[layer][crit_value]['eval_attributor']
                    # get attributions
                    attributions = attributor(
                        features, logits, gt_classes[img_idx], 0).detach().squeeze(0).squeeze(0)
                        # set img_idx to 0 bc we calculate the attributions for each image and model individually
                    pos_attr = attributions.clamp(min=0)
                    # filter bounding boxes to get the ones corresponding to filtered (single) label
                    bb_list = utils.filter_bbs(test_bbs[img_idx], gt_classes[img_idx])
                    # dilate bounding boxes if necessary
                    if args.comparison_criterion == 'dilations':
                        bb_list = utils.enlarge_bb(
                            bb_list, percentage=crit_value)
                    # create and save plots
                    ax, im = plot_contribution_map(pos_attr,
                               plot_bbs=True,
                               bb_list=bb_list)
                    plt.savefig(f'{img_path}/{layer}/attr_{crit_value}.png', bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close()
                    ax_2, im_2 = plot_contribution_map(pos_attr,
                               blended_heat_map=True,
                               #original_image=orig_image.cpu().detach().numpy().transpose(1,2,0),
                               original_image=orig_image.detach().numpy().transpose(1,2,0),
                               plot_bbs=True,
                               bb_list=bb_list)
                    plt.savefig(f'{img_path}/{layer}/attr_on_orig_{crit_value}.png', bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.close()
                print(f'Now done with evaluating injections at layer: {layer}')

parser = argparse.ArgumentParser()
parser.add_argument("--comparison_criterion", type=str, default=None, 
                    choices=['loss_functions', 'dilations', 'waterbird'], required=True, help="Name under which the main directory should be stored")
parser.add_argument("--model_backbone", type=str, choices=["bcos", "xdnn", "vanilla"], required=True, help="Model backbone to train.")
parser.add_argument("--data_path", type=str, default="datasets/", help="Path to datasets.")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["VOC2007", "COCO2014"], help="Dataset.")
parser.add_argument("--layer", type=str, default="Input", nargs='+',
                    choices=["Input", "Final", "Mid1", "Mid2", "Mid3"], help="Layer of the model to compute and optimize attributions on.")
parser.add_argument("--localization_loss_fn", type=str, default=None, nargs='+',
                    choices=["Energy", "L1", "RRR", "PPCE", "Baseline"], help="Localization loss function to use.")
parser.add_argument("--dilations", type=str, nargs='+',
                    choices=['10', '25', '50'], help='Percentage of dilation of bounding boxes')
parser.add_argument("--waterbird_settings", type=str, nargs='+',
                    choices=['conventional', 'reversed'], help='Setting of waterbird dataset.')
parser.add_argument("--attribution_method", type=str, default=None,
                    choices=["BCos", "GradCam", "IxG"], help="Attribution method(s) to use for optimization.")
parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size to use for evaluation.")
parser.add_argument("--checkpoints_path", type=str, nargs='+', default=None, help="Path to model checkpoints (if two checkpoints, then put Input checkpoint first)")
args = parser.parse_args()
main(args)