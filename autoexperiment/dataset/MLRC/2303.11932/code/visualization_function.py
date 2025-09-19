import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_contribution_map(contribution_map, ax=None, vrange=None, vmin=None, vmax=None,
                          hide_ticks=True, cmap="bwr",
                          percentile=100, 
                          blended_heat_map=False, original_image=None, alpha_overlay=0.75,
                          plot_bbs=False, bb_list=None):
    """
    Visualises a contribution map, i.e., a matrix assigning individual weights to each spatial location.
    As default, this shows a contribution map with the "bwr" colormap and chooses vmin and vmax so that the map
    ranges from (-max(abs(contribution_map), max(abs(contribution_map)).
    Args:
        contribution_map: (H, W) matrix to visualise as contributions.
        ax: axis on which to plot. If None, a new figure is created.
        vrange: If None, the colormap ranges from -v to v, with v being the maximum absolute value in the map.
            If provided, it will range from -vrange to vrange, as long as either one of the boundaries is not
            overwritten by vmin or vmax.
        vmin: Manually overwrite the minimum value for the colormap range instead of using -vrange.
        vmax: Manually overwrite the maximum value for the colormap range instead of using vrange.
        hide_ticks: Sets the axis ticks to []
        cmap: colormap to use for the contribution map plot.
        percentile: If percentile is given, this will be used as a cut-off for the attribution maps.
        blended_heat_map: Whether attribution heat map should be overlayed over grayscale version of original image
        original_image: Necessary if `blended_heat_map` is True. Expects 3D numpy image (RGB) in the shape H W C
        alpha_overlay: Alpha to set for heatmap when using `blended_heat_map` visualization mode, which overlays the
                       heat map over the greyscaled original image.
        plot_bbs: Whether to plot the bounding boxes or not
        bb_list: if plot_bbs is True, this is a list of bounding box coordinates
    Returns: The axis on which the contribution map was plotted.
    """
    assert len(
        contribution_map.shape) == 2, "Contribution map is supposed to only have spatial dimensions.."
    contribution_map = contribution_map.detach().cpu().numpy()
    cutoff = np.percentile(np.abs(contribution_map), percentile)
    contribution_map = np.clip(contribution_map, -cutoff, cutoff)
    if ax is None:
        fig, ax = plt.subplots(1)
    if vrange is None or vrange == "auto":
        vrange = max(1e-6, np.max(np.abs(contribution_map.flatten())))
    if blended_heat_map:
        assert (
                original_image is not None
            ), "Original Image expected for blended_heat_map method."
        im = ax.imshow(np.mean(original_image, axis=2), cmap="gray")
        im = ax.imshow(contribution_map, cmap=cmap,
                   vmin=-vrange if vmin is None else vmin,
                   vmax=vrange if vmax is None else vmax,
                   alpha=alpha_overlay)
    else:
        im = ax.imshow(contribution_map, cmap=cmap,
                    vmin=-vrange if vmin is None else vmin,
                    vmax=vrange if vmax is None else vmax)
    if plot_bbs:
        assert (
                bb_list is not None
        ), "List of bounding box coordinates expected for plot_bbs"
        for bb in bb_list:
                        xmin, ymin, xmax, ymax = bb
                        width = xmax - xmin
                        height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=4, edgecolor='b', facecolor='none')
                        ax.add_patch(rect)
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax, im
