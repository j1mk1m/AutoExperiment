import torch
import torchmetrics
import statistics
import torchmetrics.classification


class EnergyPointingGameBase(torchmetrics.Metric):
    """
    Base class for metrics based on EPG

    Attributes:
    include_undefined (bool): Whether to include undefined cases in metric calculation
    fractions (list): Stores the fractions of positive attributions within bounding boxes
    defined_idxs (list): Stores the indices of defined cases
    """
    def __init__(self, include_undefined=True):
        """
        Initializes the EPG instance

        Args:
        include_undefined (bool): Whether to include undefined cases in metric calculation
        """
        super().__init__()

        self.include_undefined = include_undefined

        self.add_state("fractions", default=[])
        self.add_state("defined_idxs", default=[])
        self.add_state("bbox_sizes", default=[])

    def update(self, attributions, mask_or_coords):
        raise NotImplementedError

    def compute(self):
        if len(self.fractions) == 0:
            return None
        if self.include_undefined:
            return statistics.fmean(self.fractions)
        if len(self.defined_idxs) == 0:
            return None
        return statistics.fmean([self.fractions[idx] for idx in self.defined_idxs])


class BoundingBoxEnergyMultiple(EnergyPointingGameBase):
    """
    Class that computes EPG metric for multiple bounding boxes (Inherits from EnergyPointingGameBase)
    """
    def __init__(self, include_undefined=True, min_box_size=None, max_box_size=None):
        """
        Initializes the BoundingBoxEnergyMultiple instance

        Args:
        include_undefined (bool): Whether to include undefined cases in metric calculation
        min_box_size (int): Minimum size of the bounding box to be considered
        max_box_size (int): Maximum size of the bounding box to be considered
        """
        super().__init__(include_undefined=include_undefined)
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size

    def update(self, attributions, bb_coordinates):
        """
        Updates the metric based on the provided attributions and bounding box coordinates

        Args:
        attributions (tensor): Model attributions
        bb_coordinates (list of tuples): List of bounding box coordinates
        """
        positive_attributions = attributions.clamp(min=0)
        bb_mask = torch.zeros_like(positive_attributions, dtype=torch.long)
        for coords in bb_coordinates:
            # print("coords: ", coords)
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        bb_size = len(torch.where(bb_mask == 1)[0])
        if self.min_box_size is not None and bb_size < self.min_box_size:
            return
        if self.max_box_size is not None and bb_size >= self.max_box_size:
            return
        energy_inside = positive_attributions[torch.where(bb_mask == 1)].sum()
        energy_total = positive_attributions.sum()
        assert energy_inside >= 0, energy_inside
        assert energy_total >= 0, energy_total
        if energy_total < 1e-7:
            self.fractions.append(torch.tensor(0.0))
            self.bbox_sizes.append(torch.tensor(0.0))
        else:
            self.defined_idxs.append(len(self.fractions))
            self.fractions.append(energy_inside/energy_total)
            self.bbox_sizes.append(bb_size)

class SegmentationEnergyMultiple(EnergyPointingGameBase):
    """
    Class that computes EPG metric for multiple bounding boxes (Inherits from EnergyPointingGameBase)
    """
    def __init__(self, include_undefined=True, min_box_size=None, max_box_size=None):
        """
        Initializes the BoundingBoxEnergyMultiple instance

        Args:
        include_undefined (bool): Whether to include undefined cases in metric calculation
        min_box_size (int): Minimum size of the bounding box to be considered
        max_box_size (int): Maximum size of the bounding box to be considered
        """
        super().__init__(include_undefined=include_undefined)
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size

    def update(self, attributions, bb_coordinates):
        positive_attributions = attributions.clamp(min=0)
        bb_mask = torch.zeros_like(positive_attributions, dtype=torch.long, device='cuda')
        for coords in bb_coordinates:
            mask = coords[0].squeeze()
            bb_mask[mask==1] = 1
        bb_size = len(torch.where(bb_mask == 1)[0])
        if self.min_box_size is not None and bb_size < self.min_box_size:
            return
        if self.max_box_size is not None and bb_size >= self.max_box_size:
            return
        energy_inside = positive_attributions[torch.where(bb_mask == 1)].sum()
        energy_total = positive_attributions.sum()
        assert energy_inside >= 0, energy_inside
        assert energy_total >= 0, energy_total
        if energy_total < 1e-7:
            self.fractions.append(torch.tensor(0.0))
            self.bbox_sizes.append(torch.tensor(0.0))
        else:
            self.defined_idxs.append(len(self.fractions))
            self.fractions.append(energy_inside/energy_total)
            self.bbox_sizes.append(bb_size)


class BoundingBoxIoUMultiple(EnergyPointingGameBase):
    """
    Class that implements IoU metric for attributions and bounding boxes (Inherits from EnergyPointingGameBase)
    """
    def __init__(self, include_undefined=True, iou_threshold=0.5, min_box_size=None, max_box_size=None):
        """
        Initializes the BoundingBoxIoUMultiple instance

        Args:
        include_undefined (bool): Whether to include undefined cases in metric calculation
        iou_threshold (float): Threshold for binarizing attributions in IoU calculation
        min_box_size (int): Minimum size of the bounding box to be considered
        max_box_size (int): Maximum size of the bounding box to be considered
        """
        super().__init__(include_undefined=include_undefined)
        self.iou_threshold = iou_threshold
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size

    def binarize(self, attributions):
        """
        Binarize the attributions based on a threshold

        Args:
        attributions (tensor): Model attributions

        Returns:
        tensor: Binarized attributions
        """
        attr_max = attributions.max()
        attr_min = attributions.min()
        if attr_max == 0:
            return attributions
        if torch.abs(attr_max-attr_min) < 1e-7:
            return attributions/attr_max
        return (attributions-attr_min)/(attr_max-attr_min)

    def update(self, attributions, bb_coordinates):
        positive_attributions = attributions.clamp(min=0)
        bb_mask = torch.zeros_like(positive_attributions, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        bb_size = len(torch.where(bb_mask == 1)[0])
        if self.min_box_size is not None and bb_size < self.min_box_size:
            return
        if self.max_box_size is not None and bb_size >= self.max_box_size:
            return
        binarized_attributions = self.binarize(positive_attributions)
        intersection_area = len(torch.where(
            (binarized_attributions > self.iou_threshold) & (bb_mask == 1))[0])
        union_area = len(torch.where(binarized_attributions > self.iou_threshold)[
                         0]) + len(torch.where(bb_mask == 1)[0]) - intersection_area
        assert intersection_area >= 0
        assert union_area >= 0
        if union_area == 0:
            self.fractions.append(torch.tensor(0.0))
        else:
            self.defined_idxs.append(len(self.fractions))
            self.fractions.append(torch.tensor(intersection_area/union_area))

class SegmentationIoUMultiple(EnergyPointingGameBase):
    """
    Class that implements IoU metric for attributions and bounding boxes (Inherits from EnergyPointingGameBase)
    """
    def __init__(self, include_undefined=True, iou_threshold=0.5, min_box_size=None, max_box_size=None):
        """
        Initializes the BoundingBoxIoUMultiple instance

        Args:
        include_undefined (bool): Whether to include undefined cases in metric calculation
        iou_threshold (float): Threshold for binarizing attributions in IoU calculation
        min_box_size (int): Minimum size of the bounding box to be considered
        max_box_size (int): Maximum size of the bounding box to be considered
        """
        super().__init__(include_undefined=include_undefined)
        self.iou_threshold = iou_threshold
        self.min_box_size = min_box_size
        self.max_box_size = max_box_size

    def binarize(self, attributions):
        """
        Binarize the attributions based on a threshold

        Args:
        attributions (tensor): Model attributions

        Returns:
        tensor: Binarized attributions
        """
        attr_max = attributions.max()
        attr_min = attributions.min()
        if attr_max == 0:
            return attributions
        if torch.abs(attr_max-attr_min) < 1e-7:
            return attributions/attr_max
        return (attributions-attr_min)/(attr_max-attr_min)

    def update(self, attributions, bb_coordinates):
        positive_attributions = attributions.clamp(min=0)
        bb_mask = torch.zeros_like(positive_attributions, dtype=torch.long, device='cuda')
        for coords in bb_coordinates:
            mask = coords[0].squeeze()
            bb_mask[mask==1] = 1
        bb_size = len(torch.where(bb_mask == 1)[0])
        if self.min_box_size is not None and bb_size < self.min_box_size:
            return
        if self.max_box_size is not None and bb_size >= self.max_box_size:
            return
        binarized_attributions = self.binarize(positive_attributions)
        intersection_area = len(torch.where(
            (binarized_attributions > self.iou_threshold) & (bb_mask == 1))[0])
        union_area = len(torch.where(binarized_attributions > self.iou_threshold)[
                         0]) + len(torch.where(bb_mask == 1)[0]) - intersection_area
        assert intersection_area >= 0
        assert union_area >= 0
        if union_area == 0:
            self.fractions.append(torch.tensor(0.0))
        else:
            self.defined_idxs.append(len(self.fractions))
            self.fractions.append(torch.tensor(intersection_area/union_area))
        

"""
Source: https://github.com/stevenstalder/NN-Explainer 
"""
class MultiLabelMetrics(torchmetrics.Metric):
    """
    Class for computing standard classification metrics for multi-label tasks
    """
    def __init__(self, num_classes, threshold):
        """
        Initializes the MultilabelMetrics instance

        Args:
        num_classes (int): Number of classes
        threshold (float): Threshold for classifying logits as positive or negative
        """
        super().__init__()

        self.num_classes = num_classes
        self.threshold = threshold

        self.add_state("true_positives", torch.tensor(0.0))
        self.add_state("false_positives", torch.tensor(0.0))
        self.add_state("true_negatives", torch.tensor(0.0))
        self.add_state("false_negatives", torch.tensor(0.0))

    def update(self, logits, labels):
        """
        Updates the metric counters based on model logits and labels

        Args:
        logits (tensor): Logits from the model
        labels (tensor): True labels
        """
        with torch.no_grad():
            for i, batch_sample_logits in enumerate(logits):
                for j in range(self.num_classes):
                    if labels[i][j] == 1.0:
                        if batch_sample_logits[j] >= self.threshold:
                            self.true_positives += 1.0
                        else:
                            self.false_negatives += 1.0
                    else:
                        if batch_sample_logits[j] >= self.threshold:
                            self.false_positives += 1.0
                        else:
                            self.true_negatives += 1.0

    def compute(self):
        """
        Computes the final classification metrics

        Returns:
        dict: Dictionary with following keys:
                - Accuracy
                - Precision
                - Recall
                - F-score
                - True Positives
                - True Negatives
                - False Positives
                - False Negatives
        """
        self.accuracy = ((self.true_positives + self.true_negatives) / (self.true_positives +
                         self.true_negatives + self.false_positives + self.false_negatives))
        self.precision = (self.true_positives /
                          (self.true_positives + self.false_positives))
        self.recall = (self.true_positives /
                       (self.true_positives + self.false_negatives))
        self.f_score = ((2 * self.true_positives) / (2 * self.true_positives +
                        self.false_positives + self.false_negatives))

        return {'Accuracy': self.accuracy.item(), 'Precision': self.precision.item(), 'Recall': self.recall.item(), 'F-Score': self.f_score.item(), 'True Positives': self.true_positives.item(), 'True Negatives': self.true_negatives.item(), 'False Positives': self.false_positives.item(), 'False Negatives': self.false_negatives.item()}

    def save(self, model, classifier_type, dataset):
        """
        Saves the computed metrics to a file

        Args:
        model (str): Name of the model
        classifier_type (str): Type of classifier
        dataset (str): Name of dataset
        """
        f = open(model + "_" + classifier_type + "_" +
                 dataset + "_" + "test_metrics.txt", "w")
        f.write("Accuracy: " + str(self.accuracy.item()) + "\n")
        f.write("Precision: " + str(self.precision.item()) + "\n")
        f.write("Recall: " + str(self.recall.item()) + "\n")
        f.write("F-Score: " + str(self.f_score.item()))
        f.close()


class GroupedAccuracyMetric(torchmetrics.Metric):
    def __init__(self, group_names):
        super().__init__()
        self.group_names = group_names
        self.add_state("group_correct", default=torch.zeros(len(group_names)))
        self.add_state("group_total", default=torch.zeros(len(group_names)))

    def update(self, logits, labels, groups):
        _, predictions = torch.max(logits, 1)
        # print("GROUP ACC TESTING:")
        labels = torch.argmax(labels, dim=1)
        for i, group in enumerate(groups):
            group_idx = int(group.item())
            # print("  Logits:", logits)
            # print("  Predictions:", predictions)
            # print("  Labels:", labels)
            self.group_correct[group_idx] += (predictions[i] == labels[i]).item()
            self.group_total[group_idx] += 1

    def compute(self):
        group_accuracies = {}
        for i, group_name in enumerate(self.group_names):
            if self.group_total[i] > 0:
                group_accuracies[group_name] = self.group_correct[i] / self.group_total[i]
            else:
                group_accuracies[group_name] = torch.tensor(0.0)
        return group_accuracies
    
    def save(self, model, classifier_type, dataset):
        """
        Saves the computed group accuracies to a file

        Args:
        model (str): Name of the model
        classifier_type (str): Type of classifier
        dataset (str): Name of the dataset
        """
        filename = f"{model}_{classifier_type}_{dataset}_group_accuracies.txt"
        with open(filename, "w") as f:
            for i, group_name in enumerate(self.group_names):
                if self.group_total[i] > 0:
                    accuracy = self.group_correct[i] / self.group_total[i]
                else:
                    accuracy = 0.0  # Default to 0 if no data for group
                f.write(f"{group_name}: {accuracy:.4f}\n")
