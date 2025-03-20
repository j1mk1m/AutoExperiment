import torch
import random
import numpy as np
import copy
from collections import OrderedDict
import os


def remove_module(state_dict):
    """
    Removes the 'module.' prefix from each key in the state dict of a model

    Used when loading a model state dict saved from a model wrapped in
    nn.DataParallel, which adds 'module.' prefix to every key

    Args:
    state_dict (OrderedDict): State dict of the model

    Returns:
    OrderedDict: Same dict as input but with 'module.' prefix removed
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict


def set_seed(seed):
    """
    Sets the seed for random number generation
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def filter_bbs(bb_coordinates, gt, waterbirds=False,seg=False):
    """
    Filters bounding box coordinates based on ground truth class

    Args:
    bb_coordinates (list of tuples): List of bounding box coordinates
    gt (int): Ground truth class used to filter bounding boxes
    """
    if waterbirds:
        # bb_list = []
        # print("TEST FILTER BBS:")
        # print("bb_coordinates:", bb_coordinates)
        if bb_coordinates[0] == gt:
            bb_list = [bb_coordinates[1:].long()]
            # bb_list.append(bb_coordinates[1:])
            # print("bb_list:", bb_list)
        return bb_list
    elif seg:
        bb_coordinates = bb_coordinates.squeeze().cuda(device='cuda:1')
        mask = torch.zeros_like(bb_coordinates, device='cuda:1')
        mask[bb_coordinates==(gt+1)]=1
        return mask
    else:
        bb_list = []
        # print("TEST FILTER BBS:")
        # print("bb_coordinates:", bb_coordinates)
        for bb in bb_coordinates:
            # print(" bb:", bb)
            # print(" bb.shape:", bb.shape)
            # print(" bb[0]:", bb[0])
            # print("  gt:", gt)
            if bb[0] == gt:
                bb_list.append(bb[1:])
        return bb_list


class BestMetricTracker:
    """
    Tracks best metric value achieved during training and the corresponding model state
    """
    def __init__(self, metric_name):
        """
        Initializes the BestMetricTracker instance

        Args:
        metric_name (str): Name of the metric to track
        """
        super().__init__()
        self.metric_name = metric_name
        self.best_model_dict = None
        self.best_epoch = None
        self.best_metrics = None
        self.initialized = False

    def update_values(self, metric_dict, model, epoch):
        """
        Updates the best metric value, model state dict, and the corresponding epoch

        Args:
        metric_dict (dict): Dictionary containing current metrics
        model (torch.nn.Module): Current state of the model
        epoch (int): Current epoch number
        """
        self.best_model_dict = copy.deepcopy(model.state_dict())
        self.best_metrics = copy.deepcopy(metric_dict)
        self.best_epoch = epoch

    def update(self, metric_dict, model, epoch):
        """
        Compares current metrics with the best recorded metrics and updates if better

        Args:
        metric_dict (dict): Dictionary containing current metrics
        model (torch.nn.Module): Current state of the model
        epoch (int): Current epoch number
        """
        if not self.initialized:
            self.update_values(metric_dict, model, epoch)
            self.initialized = True
        elif self.best_metrics[self.metric_name] < metric_dict[self.metric_name]:
            self.update_values(metric_dict, model, epoch)

    def get_best(self):
        """
        Returns best metric value, correpsonding model state dict, epoch number, and all metrics at the best epoch

        Returns:
        tuple: Tuple with values described above or None for all elements if no update was made
        """
        if not self.initialized:
            return None, None, None, None
        return self.best_metrics[self.metric_name], self.best_model_dict, self.best_epoch, self.best_metrics



def get_random_optimization_targets(targets):
    """
    Randomly selects targets for optimization based on their probabilities

    Args:
    targets (tensor): Tensor of target probabilities

    Returns:
    tensor: Tensor of randomly selected targets
    """
    probabilities = targets/targets.sum(dim=1, keepdim=True).detach()
    return probabilities.multinomial(num_samples=1).squeeze(1)


class ParetoFrontModels:

    def __init__(self, bin_width=0.005):
        super().__init__()
        self.bin_width = bin_width
        self.pareto_checkpoints = []
        self.pareto_costs = []

    def update(self, model, metric_dict, epoch):
        metric_vals = copy.deepcopy(metric_dict)
        state_dict = copy.deepcopy(model.state_dict())
        metric_vals.update({"model": state_dict, "epochs": epoch+1})
        self.pareto_checkpoints.append(metric_vals)
        self.pareto_costs.append(
            [metric_vals["F-Score"], metric_vals["BB-Loc"], metric_vals["BB-IoU"]])
        efficient_indices = self.is_pareto_efficient(
            -np.round(np.array(self.pareto_costs) / self.bin_width, 0)*self.bin_width, return_mask=False)
        self.pareto_checkpoints = [
            self.pareto_checkpoints[idx] for idx in efficient_indices]
        self.pareto_costs = [self.pareto_costs[idx]
                             for idx in efficient_indices]
        print(f"Current Pareto Front Size: {len(self.pareto_checkpoints)}")
        pareto_str = ""
        for idx, cost in enumerate(self.pareto_costs):
            pareto_str += f"({cost[0]:.4f},{cost[1]:.4f},{cost[2]:.4f},{self.pareto_checkpoints[idx]['epochs']})"
        print(f"Pareto Costs: {pareto_str}")

    def get_pareto_front(self):
        return self.pareto_checkpoints, self.pareto_costs

    def save_pareto_front(self, save_path):
        augmented_path = os.path.join(save_path, "pareto_front")
        os.makedirs(augmented_path, exist_ok=True)
        for idx in range(len(self.pareto_checkpoints)):
            f_score = self.pareto_checkpoints[idx]["F-Score"]
            bb_score = self.pareto_checkpoints[idx]["BB-Loc"]
            iou_score = self.pareto_checkpoints[idx]["BB-IoU"]
            epoch = self.pareto_checkpoints[idx]["epochs"]
            torch.save(self.pareto_checkpoints[idx], os.path.join(
                augmented_path, f"model_checkpoint_pareto_{f_score:.4f}_{bb_score:.4f}_{iou_score:.4f}_{epoch}.pt"))

    def is_pareto_efficient(self, costs, return_mask=True):
        """
        Find the pareto-efficient points
        : param costs: An(n_points, n_costs) array
        : param return_mask: True to return a mask
        : return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an(n_points, ) boolean array
            Otherwise it will be a(n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(
                costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            # Remove dominated points
            is_efficient = is_efficient[nondominated_point_mask]
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(
                nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient


def enlarge_bb(bb_list, percentage=0):
    """
    Enlarges each bounding box in the list by a specified percentage

    Args:
    bb_list (list of lists): List containing bounding box coordinates. Each bounding box represented as [xmin, ymin, xmax, ymax]
    percentage (float): Percentage by which to enlarge the bounding boxes (default 0 means no enlargement)

    Returns:
    list of lists: List containing the enlarged bounding boxes 
    """
    en_bb_list = []
    for bb_coord in bb_list:
        xmin, ymin, xmax, ymax = bb_coord
        width = xmax - xmin
        height = ymax - ymin
        w_margin = int(percentage * width)
        h_margin = int(percentage * height)
        new_xmin = max(0, xmin-w_margin)
        new_xmax = min(223, xmax+w_margin)
        new_ymin = max(0, ymin-h_margin)
        new_ymax = min(223, ymax+h_margin)
        en_bb_list.append([new_xmin, new_ymin, new_xmax, new_ymax])
    return en_bb_list


def update_val_metrics(metric_vals):
    """
    Renames keys of validation metrics for clarity

    Args:
    metric_vals (dict0): Dictionary containing metric values

    Returns:
    dict: Same dict but keys have prefix 'Val-'
    """
    metric_vals["Val-Accuracy"] = metric_vals.pop("Accuracy")
    metric_vals["Val-Precision"] = metric_vals.pop("Precision")
    metric_vals["Val-Recall"] = metric_vals.pop("Recall")
    metric_vals["Val-F-Score"] = metric_vals.pop("F-Score")
    metric_vals["Val-Average-Loss"] = metric_vals.pop("Average-Loss")
    if "BB-Loc" in metric_vals:
        metric_vals["Val-BB-Loc"] = metric_vals.pop("BB-Loc")
        metric_vals["Val-BB-IoU"] = metric_vals.pop("BB-IoU")
    return metric_vals

def get_bb_area(bb_list):
    """
    Returns the area of each bounding box in the list

    Args:
    bb_list (list of lists): List containing bounding box coordinates. Each bounding box represented as [xmin, ymin, xmax, ymax]

    Returns:
    list: List containing the area of each bounding box
    """
    area = 0.0
    for bb_coord in bb_list:
        xmin, ymin, xmax, ymax = bb_coord
        width = xmax - xmin
        height = ymax - ymin
        area += width*height
    return area