import torch
import captum
import captum.attr


def get_attributor(model, attributor_name, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
    """
    Function to create an attributor based on the given name

    Args:
    model (torch.nn.Module): Model for which to compute the attributions
    attributor_name (str): Name of the attribution method (BCos, GradCam or IxG)
    only_positive (bool, optional): If True, keep only positive attributions
    binarize (bool, optional): If True, binarize attributions
    interpolate (bool, optional): If True, interpolate attributions to the specified size
    interpolate_dims (tuple, optional): Target dimensions for interpolation
    batch_mode (bool, optional): If True, process inputs in batches

    Returns:
    An instance of the specified Attributor class
    """
    attributor_map = {
        "BCos": BCosAttributor,
        "GradCam": GradCamAttributor,
        "IxG": IxGAttributor
    }
    return attributor_map[attributor_name](model, only_positive, binarize, interpolate, interpolate_dims, batch_mode)


class AttributorBase:
    """
    Base class for different attribution methods

    Attributes:
    model (torch.nn.Module): Model for which the attributions are computed
    only_positive (bool): If True, keep only positive attributions
    binarize (bool): If True, binarize attributions
    interpolate (bool): If True interpolate attributions to the speficied size
    interpolate_dims (tuple): Target dimensions for interpolation
    batch_mode (bool): If True, process inputs in batches
    """
    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        """
        Initializes the AttributorBase with model and configuration settings
        
        Args:
        model (torch.nn.Module): Model for which the attributions are computed
        only_positive (bool): If True, keep only positive attributions
        binarize (bool): If True, binarize attributions
        interpolate (bool): If True interpolate attributions to the speficied size
        interpolate_dims (tuple): Target dimensions for interpolation
        batch_mode (bool): If True, process inputs in batches
        """
        super().__init__()
        self.model = model
        self.only_positive = only_positive
        self.binarize = binarize
        self.interpolate = interpolate
        self.interpolate_dims = interpolate_dims
        self.batch_mode = batch_mode

    def __call__(self, feature, output, class_idx=None, img_idx=None, classes=None):
        """
        Compute attributions for the given input

        Args:
        feature (tensor): Input features
        output (tensor): Output of the model
        class_idx (int, optional): Index of the target class for which attributions are computed. Required if batch_mode=False
        img_idx (int, optional): Index of the image in the batch. Required if batch_mode=False
        classes (tensor, optional): Tensor of class indices for each image in the batch. Required if batch_mode=True

        Returns:
        tensor: Computed attributions
        """
        if self.batch_mode:
            return self._call_batch_mode(feature, output, classes)
        return self._call_single(feature, output, class_idx, img_idx)

    def _call_batch_mode(self, feature, output, classes):
        """
        Compute attributions in batch mode?
        """
        raise NotImplementedError

    def _call_single(self, feature, output, class_idx, img_idx):
        """
        Compute attributions for a single example?
        """
        raise NotImplementedError

    def check_interpolate(self, attributions):
        """
        Interpolates attributions if interpolation is enabled

        Args:
        attributions (tensor): Computed attributions

        Returns:
        tensor: Interpolated attributions
        """
        if self.interpolate:
            return captum.attr.LayerAttribution.interpolate(
                attributions, interpolate_dims=self.interpolate_dims, interpolate_mode="bilinear")
        return attributions

    def check_binarize(self, attributions):  # normalize attributions into the range between 0 and 1
        """
        Binarize attributions if binarization is enabled

        Args:
        attributions (tensor): Computed attributions

        Returns:
        tensor: Binarized attributions
        """
        if self.binarize:
            attr_max = attributions.abs().amax(dim=(1, 2, 3), keepdim=True)
            attributions = torch.where(
                attr_max == 0, attributions, attributions/attr_max)
        return attributions

    def check_only_positive(self, attributions):
        """
        Keeps only positive attributions if enabled

        Args:
        attributions (tensor): Computed attributions

        Returns:
        tensor: Attributions with only positive values
        """
        if self.only_positive:
            return attributions.clamp(min=0)
        return attributions

    def apply_post_processing(self, attributions):
        """
        Applies post-processing steps to attributions
        
        Post processing steps:
          - Keeping only positive values
          - Binarization
          - Interpolation

        Args:
        attributions (tensor): Computed attributions

        Returns:
        tensor: Post-processed attributions
        """
        attributions = self.check_only_positive(attributions)
        attributions = self.check_binarize(attributions)
        attributions = self.check_interpolate(attributions)
        return attributions


class BCosAttributor(AttributorBase):
    """
    BCos attribution method implementation

    Attributes:
    Inherits all attributes from AttributorBase
    """
    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        """
        Initilaizes BCosAttributor with the model and configuration settings

        Args:
        Same as in the initializer of the AttributorBase class
        """
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        with self.model.explanation_mode():
            grads = torch.autograd.grad(torch.unbind(
                target_outputs), feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads*feature).sum(dim=1, keepdim=True)
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx):
        with self.model.explanation_mode():
            grads = torch.autograd.grad(
                output[img_idx, class_idx], feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads[img_idx]*feature[img_idx]
                        ).sum(dim=0, keepdim=True).unsqueeze(0)
        return self.apply_post_processing(attributions)


class GradCamAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        grads = torch.autograd.grad(torch.unbind(
            target_outputs), feature, create_graph=True, retain_graph=True)[0]
        grads = grads.mean(dim=(2, 3), keepdim=True)
        prods = grads * feature
        attributions = torch.nn.functional.relu(
            torch.sum(prods, axis=1, keepdim=True))
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx):
        grads = torch.autograd.grad(
            output[img_idx, class_idx], feature, create_graph=True, retain_graph=True)[0]
        grads = grads.mean(dim=(2, 3), keepdim=True)
        prods = grads[img_idx] * feature[img_idx]
        attributions = torch.nn.functional.relu(
            torch.sum(prods, axis=0, keepdim=True)).unsqueeze(0)
        return self.apply_post_processing(attributions)


class IxGAttributor(AttributorBase):

    def __init__(self, model, only_positive=False, binarize=False, interpolate=False, interpolate_dims=(224, 224), batch_mode=False):
        super().__init__(model=model, only_positive=only_positive, binarize=binarize,
                         interpolate=interpolate, interpolate_dims=interpolate_dims, batch_mode=batch_mode)

    def _call_batch_mode(self, feature, output, classes):
        target_outputs = torch.gather(output, 1, classes.unsqueeze(-1))
        grads = torch.autograd.grad(torch.unbind(
            target_outputs), feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads * feature).sum(dim=1, keepdim=True)
        return self.apply_post_processing(attributions)

    def _call_single(self, feature, output, class_idx, img_idx):
        grads = torch.autograd.grad(
            output[img_idx, class_idx], feature, create_graph=True, retain_graph=True)[0]
        attributions = (grads[img_idx] * feature[img_idx]
                        ).sum(dim=0, keepdim=True).unsqueeze(0)
        return self.apply_post_processing(attributions)