import torch
import os
import argparse
import torchvision
from tqdm import tqdm
import datasets
import argparse
import torch.utils.tensorboard
import utils
import copy
import losses
import metrics
import bcos.models
import model_activators
import attribution_methods
import hubconf
import bcos
import bcos.modules
import bcos.data
import fixup_resnet
import json

def eval_model(model, attributor, loader, num_batches, num_classes, loss_fn, writer=None, epoch=None):
    """
    Evaluate the model using the provided data loader and compute various metrics.

    Args:
    model (torch.nn.Module): The model to be evaluated
    attributor (Callable): Function to compute attributions for model explanations
    loader (torch.utils.data.DataLoader): DataLoader for evaluation data
    num_batches (int): Number of batches in the DataLoader
    num_classes (int): Number of classes in the dataset
    loss_fn (Callable): Loss function used for evaluation
    writer (optional): Tensorboard writer for logging metrics
    epoch (int, optional): Current epoch number for logging

    Returns:
    dict: Dictionary containing the following eval metrics: F1, BB, IoU
    """
    model.eval()
    f1_metric = metrics.MultiLabelMetrics(
        num_classes=num_classes, threshold=0.0)
    bb_metric = metrics.BoundingBoxEnergyMultiple()
    iou_metric = metrics.BoundingBoxIoUMultiple()
    total_loss = 0
    for batch_idx, (test_X, test_y, test_bbs) in enumerate(tqdm(loader)):
        test_X.requires_grad = True
        test_X = test_X.cuda()
        test_y = test_y.cuda()
        logits, features = model(test_X)
        loss = loss_fn(logits, test_y).detach()
        total_loss += loss
        f1_metric.update(logits, test_y)

        if attributor:
            for img_idx in range(len(test_X)):
                class_target = torch.where(test_y[img_idx] == 1)[0]
                for pred_idx, pred in enumerate(class_target):
                    attributions = attributor(
                        features, logits, pred, img_idx).detach().squeeze(0).squeeze(0)
                    bb_list = utils.filter_bbs(test_bbs[img_idx], pred)
                    bb_metric.update(attributions, bb_list)
                    iou_metric.update(attributions, bb_list)

    metric_vals = f1_metric.compute()
    if attributor:
        bb_metric_vals = bb_metric.compute()
        iou_metric_vals = iou_metric.compute()
        metric_vals["BB-Loc"] = bb_metric_vals
        metric_vals["BB-IoU"] = iou_metric_vals
    metric_vals["Average-Loss"] = total_loss.item()/num_batches        
    print(f"Validation Metrics: {metric_vals}")
    model.train()
    if writer is not None:
        writer.add_scalar("val_loss", total_loss.item()/num_batches, epoch)
        writer.add_scalar("accuracy", metric_vals["Accuracy"], epoch)
        writer.add_scalar("precision", metric_vals["Precision"], epoch)
        writer.add_scalar("recall", metric_vals["Recall"], epoch)
        writer.add_scalar("fscore", metric_vals["F-Score"], epoch)
        if attributor:
            writer.add_scalar("bbloc", metric_vals["BB-Loc"], epoch)
            writer.add_scalar("bbiou", metric_vals["BB-IoU"], epoch)
    return metric_vals


def main(args):
    num_classes_dict = {"VOC2007": 20, "COCO2014":  80}
    num_classes = num_classes_dict[args.dataset]
        
    is_bcos = (args.model_backbone == "bcos")
    is_xdnn = (args.model_backbone == "xdnn")
    is_vanilla = (args.model_backbone == "vanilla")

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
    
    loss_loc = losses.get_localization_loss(
        args.localization_loss_fn) if args.localization_loss_fn else None

    loss_fn = torch.nn.BCEWithLogitsLoss()
    result_dict = dict()
    
    if 'best' in args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint["model"])
    else:
        raise FileNotFoundError
    
    layer_idx = layer_dict[args.layer]
    model = model.cuda()
    model.eval()

    model_activator = model_activators.ResNetModelActivator(
        model=model, layer=layer_idx, is_bcos=is_bcos)

    # model_prefix = args.model_backbone

    num_test_batches = len(test_data) / args.eval_batch_size

    if args.attribution_method:
        interpolate = True if layer_idx is not None else False
        eval_attributor = attribution_methods.get_attributor(
            model, args.attribution_method, loss_loc.only_positive, loss_loc.binarize, interpolate, (224, 224), batch_mode=False)
    
    # f1_best_score, f1_best_model_dict, f1_best_epoch, f1_best_metric_vals = f1_tracker.get_best()
    #f1_best_metric_vals = utils.update_val_metrics(f1_best_metric_vals)
    #model.load_state_dict(f1_best_model_dict)
    metrics = eval_model(model_activator, eval_attributor, test_loader,
                                num_test_batches, num_classes, loss_fn)
    result_dict[args.checkpoint_path] = metrics
    #f1_best_metrics.update(f1_best_metric_vals)
    #f1_best_metrics.update(
    #    {"model": f1_best_model_dict, "epochs": f1_best_epoch+1} | vars(args))
    print(f'Saving final results in result_jsons/baseline_{args.model_backbone}_{args.attribution_method}_{args.layer}_{args.localization_loss_fn}.json ...')
    with open(f'result_jsons/baseline_{args.model_backbone}_{args.attribution_method}_{args.layer}_{args.localization_loss_fn}.json', 'w') as f:
        json.dump(result_dict, f)
    print(f'Done!')

parser = argparse.ArgumentParser()
parser.add_argument("--model_backbone", type=str, choices=["bcos", "xdnn", "vanilla"], required=True, help="Model backbone to train.")

parser.add_argument("--data_path", type=str, default="datasets/", help="Path to datasets.")
parser.add_argument("--dataset", type=str, required=True,
                    choices=["VOC2007", "COCO2014"], help="Dataset to train on.")
parser.add_argument("--layer", type=str, default="Input",
                    choices=["Input", "Final", "Mid1", "Mid2", "Mid3"], help="Layer of the model to compute and optimize attributions on.")
parser.add_argument("--localization_loss_fn", type=str, default=None,
                    choices=["Energy", "L1", "RRR", "PPCE"], help="Localization loss function to use.")
parser.add_argument("--attribution_method", type=str, default=None,
                    choices=["BCos", "GradCam", "IxG"], help="Attribution method to use for optimization.")
parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size to use for evaluation.")
parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to pareto-dominant models")
args = parser.parse_args()
main(args)
        
