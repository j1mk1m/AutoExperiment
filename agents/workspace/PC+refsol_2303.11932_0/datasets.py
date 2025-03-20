import numpy as np
import os
import torch
import utils


class VOCDetectParsed(torch.utils.data.Dataset):
    """
    Dataset class for PASCAL VOC2007 detection dataset

    Attributes:
    data (tensor): Tensor containing image data
    labels (tensor): Tensor containing labels for each image
    bbs (list): List of bounding boxes for each image
    annotated_fraction (float): Fraction of the dataset to consider as annotated

    Methods:
    __init__: Initializes dataset object, loads the data, applies settings
    __getitem__: Returns a single item from the dataset
    __len__: Returns the length of the dataset
    load_data: Loads data, label, and bounding box for a specified image and class
    collate_fn: Custom collate function for data loading
    """
    def __init__(self, root, image_set, transform=None, annotated_fraction=1.0):
        """
        Initializes the dataset object

        Args:
        root (str): Root dir where the data is stored
        image_set (str): Whether to use 'train', 'val' or 'test' subset of the data
        transform (callable, optional): Function that determines what transformation to apply to the images
        annotated_fraction (float): Fraction of the dataset to consider as annotated
        """
        super().__init__()
        # Load dataset from specified file
        data_dict = torch.load(os.path.join(root, image_set + ".pt"))
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]
        self.bbs = data_dict["bbs"]

        # Make sure length of data, labels, and bbs is matching
        assert len(self.data) == len(self.labels)
        assert len(self.data) == len(self.bbs)
        self.annotated_fraction = annotated_fraction
        
        # If only a fraction of data is annotated, randomly select that fraction
        if self.annotated_fraction < 1.0:
            annotated_indices = np.random.choice(len(self.bbs), size=int(
                self.annotated_fraction*len(self.bbs)), replace=False)
            # Set bounding boxes to None for unannotated data
            for bb_idx in range(len(self.bbs)):
                if bb_idx not in annotated_indices:
                    self.bbs[bb_idx] = None
        self.transform = transform

    def __getitem__(self, idx):
        """
        Return an item by index

        Args:
        idx (int): Index of the item

        Returns:
        tuple: (image, label, bounding box) where the image is the transformed image if transform is provided
        """
        if self.transform is not None:
            return self.transform(self.data[idx]), self.labels[idx], self.bbs[idx]
        return self.data[idx], self.labels[idx], self.bbs[idx]

    def load_data(self, idx, pred_class):
        """
        Load data, label, and bounding box for a specific image and predicted class

        Args:
        idx (int): Index of the image in the dataset
        pred_class (int): The class for which to filter the bounding box

        Returns:
        tuple: (image, label, filtered bounding box)
        """
        img, labels, bbs = self.__getitem__(idx)
        label = labels[pred_class]
        bb = utils.filter_bbs(bbs, pred_class)
        return img, label, bb

    def __len__(self):
        """
        Return the length of the dataset

        Returns:
        int: Number of items in the dataset
        """
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        """
        Function that combines a list of samples into a mini-batch

        Args:
        batch (list of tuples): a list where each tuple consists of (data, labels, bounding boxes) for a single sample

        Returns:
        tuple: A tuple containing 3 elements:
                - data (tensor): batch of images combined into a single tensor
                - labels (tensor): batch of labels combined into a single tensor
                - bbs (list of lists): a list where each sublist contains bounding boxes corresponding to each img in the batch
        """
        # Combine img tensors from each sample into a single batch tensor
        data = torch.stack([item[0] for item in batch])
        
        # Combine label tensors from each sample into a single batch tensor
        labels = torch.stack([item[1] for item in batch])

        # Combine the bounding boxes from each sample into a list of lists
        bbs = [item[2] for item in batch]

        return data, labels, bbs

class VOCDetectSegParsed(torch.utils.data.Dataset):
    """
    Dataset class for PASCAL VOC2007 detection dataset

    Attributes:
    data (tensor): Tensor containing image data
    labels (tensor): Tensor containing labels for each image
    bbs (list): List of bounding boxes for each image
    annotated_fraction (float): Fraction of the dataset to consider as annotated

    Methods:
    __init__: Initializes dataset object, loads the data, applies settings
    __getitem__: Returns a single item from the dataset
    __len__: Returns the length of the dataset
    load_data: Loads data, label, and bounding box for a specified image and class
    collate_fn: Custom collate function for data loading
    """
    def __init__(self, root, image_set, transform=None, annotated_fraction=1.0):
        """
        Initializes the dataset object

        Args:
        root (str): Root dir where the data is stored
        image_set (str): Whether to use 'train', 'val' or 'test' subset of the data
        transform (callable, optional): Function that determines what transformation to apply to the images
        annotated_fraction (float): Fraction of the dataset to consider as annotated
        """
        super().__init__()
        # Load dataset from specified file
        data_dict = torch.load(os.path.join(root, image_set + ".pt"))
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]
        self.bbs = data_dict["mask"]

        # Make sure length of data, labels, and bbs is matching
        assert len(self.data) == len(self.labels)
        assert len(self.data) == len(self.bbs)
        self.annotated_fraction = annotated_fraction
        
        # If only a fraction of data is annotated, randomly select that fraction
        if self.annotated_fraction < 1.0:
            annotated_indices = np.random.choice(len(self.bbs), size=int(
                self.annotated_fraction*len(self.bbs)), replace=False)
            # Set bounding boxes to None for unannotated data
            for bb_idx in range(len(self.bbs)):
                if bb_idx not in annotated_indices:
                    self.bbs[bb_idx] = None
        self.transform = transform

    def __getitem__(self, idx):
        """
        Return an item by index

        Args:
        idx (int): Index of the item

        Returns:
        tuple: (image, label, bounding box) where the image is the transformed image if transform is provided
        """
        if self.transform is not None:
            return self.transform(self.data[idx]), self.labels[idx], self.bbs[idx]
        return self.data[idx], self.labels[idx], self.bbs[idx]

    def load_data(self, idx, pred_class):
        """
        Load data, label, and bounding box for a specific image and predicted class

        Args:
        idx (int): Index of the image in the dataset
        pred_class (int): The class for which to filter the bounding box

        Returns:
        tuple: (image, label, filtered bounding box)
        """
        img, labels, bbs = self.__getitem__(idx)
        label = labels[pred_class]
        bb = utils.filter_bbs(bbs, pred_class)
        return img, label, bb

    def __len__(self):
        """
        Return the length of the dataset

        Returns:
        int: Number of items in the dataset
        """
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        """
        Function that combines a list of samples into a mini-batch

        Args:
        batch (list of tuples): a list where each tuple consists of (data, labels, bounding boxes) for a single sample

        Returns:
        tuple: A tuple containing 3 elements:
                - data (tensor): batch of images combined into a single tensor
                - labels (tensor): batch of labels combined into a single tensor
                - bbs (list of lists): a list where each sublist contains bounding boxes corresponding to each img in the batch
        """
        # Combine img tensors from each sample into a single batch tensor
        data = torch.stack([item[0] for item in batch])
        
        # Combine label tensors from each sample into a single batch tensor
        labels = torch.stack([item[1] for item in batch])

        # Combine the bounding boxes from each sample into a list of lists
        bbs = [item[2] for item in batch]

        return data, labels, bbs

class WaterbirdsDetectParsed(torch.utils.data.Dataset):
    """
    Dataset class for the Waterbirsd-100 dataset structured as a list of dictionaries.
    Each dictionary contains keys for 'image', 'label', 'group', 'bbox', and others.

    Attributes:
    dataset (list): List of dictionaries containing dataset information.
    transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
    annotated_fraction (float): Fraction of the dataset to consider as annotated.

    Methods:
    __init__: Initializes dataset object, loads the data.
    __getitem__: Returns a single item from the dataset.
    __len__: Returns the length of the dataset.
    """
    def __init__(self, root, image_set, transform=None, annotated_fraction=1.0):
        """
        Initializes the dataset object.

        Args:
        root (str): Root directory where the data is stored.
        image_set (str): Specifies the subset ('train', 'val', 'test') of the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        annotated_fraction (float): Fraction of the dataset to consider as annotated.
        """
        super().__init__()
        self.dataset = torch.load(os.path.join(root, f"{image_set}.pt"))
        self.transform = transform
        self.annotated_fraction = annotated_fraction
        
        if self.annotated_fraction < 1.0:
            total_annotations = int(len(self.dataset) * self.annotated_fraction)
            annotated_indices = np.random.choice(len(self.dataset), size=total_annotations, replace=False)
            self.annotated_indices = set(annotated_indices)
        else:
            self.annotated_indices = set(range(len(self.dataset)))

    def __getitem__(self, idx):
        """
        Returns a single item from the dataset.

        Args:
        idx (int): Index of the item.

        Returns:
        tuple: (image, label, bbox) where image is the transformed image if a transform is provided,
               label is the label of the image, and bbox is the bounding box if available.
        """
        data_point = self.dataset[idx]
        image = data_point['image']  # Assuming 'image' is already a tensor
        
        label = data_point['label']
        label_one_hot = torch.zeros(2)
        label_one_hot[int(label)] = 1

        bbox = data_point['bbox_coords'] if idx in self.annotated_indices else None

        group = data_point['group']

        if self.transform:
            image = self.transform(image)

        return image, label_one_hot, bbox, group
    
    def load_data(self, idx, pred_class):
        """
        Load data, label, and bounding box for a specific image and predicted class.

        Args:
        idx (int): Index of the image in the dataset.
        pred_class (int): The class for which to filter the bounding box.

        Returns:
        tuple: (image, label, filtered bounding box)
        """
        img, labels, bbox, group = self.__getitem__(idx)
        label = labels[pred_class]
        if bbox is not None:
            # Filtering bounding boxes for the predicted class
            filtered_bbox = utils.filter_bbs(bbox, pred_class)
        else:
            filtered_bbox = None
        return img, label, filtered_bbox, group

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: Total number of items in the dataset.
        """
        return len(self.dataset)

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to combine a list of samples into a mini-batch.

        Args:
        batch (list of tuples): A list where each tuple contains (image, label, bbox).

        Returns:
        tuple: Combined batch of images, labels, and bboxes.
        """
        # Extract separate lists of images, labels, and bboxes from the batch
        images, labels, bboxes, groups = zip(*batch)
        
        # Stack images and labels to form tensors
        images = torch.stack(images)
        labels = torch.stack(labels)
        
        return images, labels, bboxes, groups

