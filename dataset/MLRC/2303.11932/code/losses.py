import torch


def get_localization_loss(loss_name):
    """
    Creates localization loss object based on the name

    Args:
    loss_name (str): Name of the desired loss (Energy, L1, RRR or PPCE)

    Returns:
    An instance of the specified loss class
    """
    loss_map = {
        "Energy": EnergyPointingGameBBMultipleLoss,
        "Energy_seg": EnergyPointingGameSegMultipleLoss,
        "L1": GradiaBBMultipleLoss,
        "RRR": RRRBBMultipleLoss,
        "PPCE": HAICSBBMultipleLoss
    }
    return loss_map[loss_name]()


class BBMultipleLoss:
    """
    Base class for bounding box based multiple loss calculations
    """
    def __init__(self):
        """
        Initializes the BBMultpipleLoss instance
        """
        super().__init__()

    def __call__(self, attributions, bb_coordinates):
        raise NotImplementedError

    def get_bb_mask(self, bb_coordinates, mask_shape):
        """
        Creates a binary mask from bounding box coordinates

        Args:
        bb_coordinates (list of tuples): List of bounding box coordinates
        mask_shape (tuple): The shape of the mask to be created

        Returns:
        tensor: Binary mask with the same shape as mask_shape
        """
        bb_mask = torch.zeros(mask_shape, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        return bb_mask


class EnergyPointingGameBBMultipleLoss:
    """
    Class implementing Energy (EPG based) loss for bounding boxes
    """
    def __init__(self):
        """
        Initialize an instance of EnergyPointingGameBBMultipleLoss
        """
        super().__init__()
        self.only_positive = False
        self.binarize = False

    def __call__(self, attributions, bb_coordinates):
        """
        Compute the Energy loss

        Args:
        attributions (tensor): Attributions from the model
        bb_coordinates (list of tuples): List of bounding box coordinates

        Returns:
        float: Computed loss
        """
        pos_attributions = attributions.clamp(min=0)
        bb_mask = torch.zeros_like(pos_attributions, dtype=torch.long)
        for coords in bb_coordinates:
            xmin, ymin, xmax, ymax = coords
            bb_mask[ymin:ymax, xmin:xmax] = 1
        num = pos_attributions[torch.where(bb_mask == 1)].sum()
        den = pos_attributions.sum()
        if den < 1e-7:
            return 1-num
        return 1-num/den


class RRRBBMultipleLoss(BBMultipleLoss):
    """
    Implements RRR localization loss
    """
    def __init__(self):
        super().__init__()
        self.only_positive = False
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape)
        irrelevant_attrs = attributions[torch.where(bb_mask == 0)]
        return torch.square(irrelevant_attrs).sum()


class GradiaBBMultipleLoss(BBMultipleLoss):
    """
    Implements L1 localization loss
    """
    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.only_positive = True
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape).cuda()
        return self.l1_loss(attributions, bb_mask)


class HAICSBBMultipleLoss(BBMultipleLoss):
    """
    Implements PPCE localization loss
    """
    def __init__(self):
        super().__init__()
        self.bce_loss = torch.nn.BCELoss(reduction='mean')
        self.only_positive = True
        self.binarize = True

    def __call__(self, attributions, bb_coordinates):
        bb_mask = self.get_bb_mask(bb_coordinates, attributions.shape)
        attributions_in_box = attributions[torch.where(bb_mask == 1)]
        return self.bce_loss(attributions_in_box, torch.ones_like(attributions_in_box))


class EnergyPointingGameSegMultipleLoss:
    """
    Class implementing Energy (EPG based) loss for bounding boxes
    """
    def __init__(self):
        """
        Initialize an instance of EnergyPointingGameBBMultipleLoss
        """
        super().__init__()
        self.only_positive = False
        self.binarize = False

    def __call__(self, attributions, bb_coordinates):
        """
        Compute the Energy loss

        Args:
        attributions (tensor): Attributions from the model
        bb_coordinates (list of tuples): List of bounding box coordinates

        Returns:
        float: Computed loss
        """
        pos_attributions = attributions.clamp(min=0)
        num = pos_attributions[torch.where(bb_coordinates.cuda(device='cuda:1') == 1)].sum()
        den = pos_attributions.sum()
        if den < 1e-7:
            return 1-num
        return 1-num/den
