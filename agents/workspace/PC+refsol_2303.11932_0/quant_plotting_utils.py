import json
import os
import numpy as np
import matplotlib.pyplot as plt


def pareto_plot(pareto_front_data, baseline_data, score_type='BB-IoU', filter_front=True, show_layer=None):
    configurations = [
        ("vanilla", "IxG", "Input"),
        ("xdnn", "IxG", "Input"),
        ("bcos", "BCos", "Input"),
        ("vanilla", "IxG", "Final"),
        ("xdnn", "IxG", "Final"),
        ("bcos", "BCos", "Final")]

    model_names = {
        "vanilla": "Vanilla ResNet-50",
        "xdnn": "X-DNN ResNet-50",
        "bcos": "B-cos ResNet-50"}

    attribution_titles = ["IxG", "IntGrad", "B-cos"]

    # Filter configurations based on show_layer
    if show_layer is not None:
        configurations = [
            config for config in configurations if show_layer in config]

    plt.rcParams['font.family'] = 'serif'
    markers = {"baseline": "X", "Energy": "o",
               "L1": "v", "PPCE": "p", "RRR*": "D"}
    colours = {"baseline": "#FFFFFF", "Energy": "#ED3E63",
               "L1": "#39D095", "PPCE": "#FECB5B", "RRR*": "#187FA8"}

    x_lim = find_limits("F-Score", baseline_data)
    y_lim = find_limits(score_type, pareto_front_data)

    # Adjust subplot creation based on the number of filtered configurations
    num_plots = len(configurations)
    num_rows = 1 if show_layer is not None else 2
    num_cols = num_plots // num_rows

    _, axes = plt.subplots(num_rows, num_cols, figsize=(
        18, 5 if show_layer is None else 2.5), dpi=300, sharex='col', sharey='row')
    # Ensure axes is a flat array for easy iteration
    axes = np.array(axes).reshape(-1)

    for idx, (ax, config) in enumerate(zip(axes, configurations)):
        model_name, _, layer = config

        if score_type == "BB-Loc":
            ax.set_xlim(55, 75)
            ax.set_ylim(20, 58)
        elif score_type == "BB-IoU":
            ax.set_xlim(40, 75)
            ax.set_ylim(0, 60)

        ax.set_xticks(
            np.arange(ax.get_xlim()[0] // 5 * 5, ax.get_xlim()[1], 5))
        ax.set_yticks(
            np.arange(ax.get_ylim()[0] // 10 * 10, ax.get_ylim()[1], 10))

        ax.set_title(model_names[model_name])

        # Add the layer labels on the far left side of the first column of subplots
        if idx % 3 == 0:
            layer_label = "Input Layer" if "Input" in layer else "Final Layer"
            ax.text(-0.2, 0.5, layer_label, transform=ax.transAxes, rotation='vertical',
                    verticalalignment='center', fontsize=14)
            ax.set_ylabel(
                "EPG Score (%)") if score_type == "BB-Loc" else ax.set_ylabel("IoU Score (%)")
        # Add x-label on the bottom row
        if idx >= 3:
            ax.set_xlabel("F1 Score (%)")

        baseline_key = config
        if baseline_key in baseline_data:
            plot_baseline(
                ax, baseline_data[baseline_key], markers, colours, score_type)

        for loss_fn in markers.keys():
            if loss_fn != "baseline":
                pareto_key = config + (loss_fn,)
                if pareto_key in pareto_front_data:
                    pareto_metrics = pareto_front_data[pareto_key]
                    pareto_points = np.array(
                        [(model["F-Score"]*100, model["BB-Loc"]*100) for model in pareto_metrics])
                    if filter_front:
                        pareto_points = find_pareto_front(pareto_points)
                    sorted_pareto_points = sort_pareto_front(pareto_points)
                    if sorted_pareto_points.size > 0:
                        f_scores, bb_locs = zip(*sorted_pareto_points)
                        ax.scatter(f_scores, bb_locs, label=loss_fn,
                                   marker=markers[loss_fn], color=colours[loss_fn], edgecolors='black', s=69, zorder=3)
                        ax.plot(f_scores, bb_locs, linestyle='--',
                                color=colours[loss_fn], linewidth=2, zorder=1)

        # Add attribution method titles
        ax.text(0.01, 0.85, attribution_titles[idx % 3], horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes, color='blue', fontsize=14)

        # Add grid
        ax.grid(True)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(
            0.5, 0.85), ncol=4, fontsize=9, frameon=True, framealpha=1)

    plt.tight_layout()
    plt.show()


def plot_dilations(plot_data, baseline_data, metric=None):
    """
    Plot the dilation results.
    Input:
    - plot_data: Data for the Pareto front
    - baseline_data: Data for the B-cos model guided at the Input layer
    - metric: The metric to plot ("EPG", "IoU", or None for both)
    """
    configurations = [
        ("Energy", "EPG"),
        ("L1", "EPG"),
        ("Energy", "IoU"),
        ("L1", "IoU")
    ]

    # Filter configurations based on the metric argument
    if metric == "EPG":
        configurations = [
            config for config in configurations if config[1] == "EPG"]
    elif metric == "IoU":
        configurations = [
            config for config in configurations if config[1] == "IoU"]

    plt.rcParams['font.family'] = 'serif'
    markers = {"baseline": "X", "Energy": "o", "L1": "v"}
    colours = {"baseline": "#FFFFFF", "0%": "#11285E",
               "10%": "#2E7AB5", "25%": "#A0C8E0", "50%": "#F6FAFF"}

    f1_lim = find_limits("F-Score", baseline_data)
    epg_lim = find_limits("BB-Loc", plot_data)
    iou_lim = find_limits("BB-IoU", plot_data)

    # Adjust the subplot layout based on the configurations length
    nrows = 1 if metric else 2
    ncols = len(configurations) // nrows

    _, axes = plt.subplots(nrows, ncols, figsize=(
        11, 4 if nrows == 2 else 2.5), dpi=300, sharex='col', sharey='row')
    if nrows * ncols == 1:  # Adjust for a single plot scenario
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (ax, config) in enumerate(zip(axes, configurations)):
        loss_fn, metric = config
        score_type = "BB-IoU" if metric == "IoU" else "BB-Loc"

        if metric == "EPG":
            ax.set_xlim(67, 71.5)
            ax.set_ylim(25, 49)
            ax.set_yticks(
                np.arange(ax.get_ylim()[0] // 10 * 10, ax.get_ylim()[1], 10))
        else:  # Adjust for IoU
            ax.set_xlim(67, 71.5)
            ax.set_ylim(0, 30)
            ax.set_yticks(
                np.arange(ax.get_ylim()[0] // 5 * 5, ax.get_ylim()[1], 5))

        ax.set_xticks(
            np.arange(ax.get_xlim()[0] // 2 * 2, ax.get_xlim()[1], 2))

        # Add loss function title on top row or if there's only one row
        if idx < ncols or nrows == 1:
            ax.set_title(f"{loss_fn} Loss", weight="bold")
        # Add y-label on the left column or if there's only one column
        if idx % ncols == 0 or ncols == 1:
            ax.set_ylabel(f"{metric} Score (%)")
        # Add x-label on the bottom row or if there's only one row
        if idx >= (nrows - 1) * ncols or nrows == 1:
            ax.set_xlabel("F1 Score (%)")

        # Plot the baseline
        plot_baseline(ax, baseline_data[(
            'bcos', 'BCos', 'Input')], markers, colours, score_type=score_type)

        for dilation in [0, 0.1, 0.25, 0.5]:
            config_key = (loss_fn, str(dilation))
            if config_key in plot_data:
                pareto_metrics = plot_data[config_key]
                pareto_points = np.array(
                    [(model["F-Score"]*100, model[score_type]*100) for model in pareto_metrics])
                sorted_pareto_points = sort_pareto_front(pareto_points)
                if sorted_pareto_points.size > 0:
                    f_scores, loc_scores = zip(*sorted_pareto_points)
                    colour_key = f"{int(dilation*100)}%"
                    ax.scatter(f_scores, loc_scores, label=colour_key,
                               marker=markers[loss_fn], color=colours[colour_key], edgecolors='black', s=69, zorder=3)

        # Add grid
        ax.grid(True)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(
            0.5, 0.835), ncol=5, fontsize=9, frameon=True, framealpha=1)

    plt.tight_layout()
    plt.show()


def plot_sparsity(plot_data, baseline_data):
    """
    Plot the sparse annotations results.
    Input:
    - plot_data: Data for the Pareto front
    - baseline_data: Data for the B-cos model guided at the Input layer
    """
    sparsities = ["0.01", "0.1", "1"]
    plt.rcParams['font.family'] = 'serif'
    markers = {"baseline": "X", "Energy": "o", "L1": "v"}
    colours = {"baseline": "#FFFFFF", "Energy": "#ED3E63", "L1": "#39D095"}

    f1_lim = find_limits("F-Score", baseline_data)
    epg_lim = find_limits("BB-Loc", plot_data)
    iou_lim = find_limits("BB-IoU", plot_data)

    _, axes = plt.subplots(1, 3, figsize=(
        12, 2.5), dpi=300, sharex='col', sharey='row')
    for idx, (ax, sparsity) in enumerate(zip(axes, sparsities)):
        ax.set_xlim(61, 72)
        ax.set_ylim(25, 65)
        ax.set_yticks(
            np.arange(ax.get_ylim()[0] // 10 * 10, ax.get_ylim()[1], 10))

        ax.set_xticks(
            np.arange(ax.get_xlim()[0] // 2 * 2, ax.get_xlim()[1], 2))

        # Add annotation percentage on top of each subplot
        ax.set_title(
            f"{int(float(sparsity) * 100)}% of annotations", weight="bold")
        # Add y-label on the left column
        if idx == 0:
            ax.set_ylabel("EPG Score (%)")
        # Add x-label on the bottom
        ax.set_xlabel("F1 Score (%)")

        # Plot the baseline
        plot_baseline(ax, baseline_data[(
            'bcos', 'BCos', 'Input')], markers, colours, score_type="BB-Loc")

        for loss_fn in ["Energy", "L1"]:
            config_key = (loss_fn, sparsity)
            if config_key in plot_data:
                pareto_metrics = plot_data[config_key]
                pareto_points = np.array(
                    [(model["F-Score"]*100, model["BB-Loc"]*100) for model in pareto_metrics])
                sorted_pareto_points = sort_pareto_front(pareto_points)
                if sorted_pareto_points.size > 0:
                    f_scores, loc_scores = zip(*sorted_pareto_points)
                    ax.scatter(f_scores, loc_scores, label=loss_fn,
                               marker=markers[loss_fn], color=colours[loss_fn], edgecolors='black', s=69, zorder=3)

        # Add grid
        ax.grid(True)

        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(
            0.5, 0.85), ncol=5, fontsize=9, frameon=True, framealpha=1)

    plt.tight_layout()
    plt.show()


def load_jsons(root="result_jsons/"):
    """Loads the data from the json files into dictionaries for the visualisation of different experiments"""
    baseline_data = {}
    lossfn_data = {}
    dilation_data = {}
    sparse_data = {}

    for model_file in os.listdir(root):
        filepath = os.path.join(root, model_file)
        if "trash" in filepath or "PPCE" in filepath:
            continue

        with open(filepath, 'r') as json_file:
            json_data = json.load(json_file)

        # Extract model configuration from the filename
        parts = model_file.split("_")

        if "baseline" in model_file:
            data_dict = baseline_data
            model = parts[1]
            attr_method = parts[2]
            layer = parts[3]
            config_key = (model, attr_method, layer)
        elif "dilation" in model_file:
            data_dict = dilation_data
            loss_fn = parts[3]
            dilation = parts[5].split(".json")[0]
            config_key = (loss_fn, dilation)
        elif "sparse" in model_file:
            data_dict = sparse_data
            loss_fn = parts[3]
            sparsity = parts[5].split(".json")[0]
            config_key = (loss_fn, sparsity)
        else:
            data_dict = lossfn_data
            model = parts[0]
            attr_method = parts[1]
            layer = parts[2]
            loss_fn = parts[3] if len(parts) > 4 else parts[3].split(".")[0]
            config_key = (model, attr_method, layer, loss_fn)

        # Initialise a list for this model if it doesn't exist yet
        if config_key not in data_dict:
            data_dict[config_key] = []

        # Add the metrics for this checkpoint to the list
        for checkpoint_name, metrics in json_data.items():
            metrics_dict = {
                "F-Score": metrics["F-Score"],
                "BB-Loc": metrics["BB-Loc"],
                "BB-IoU": metrics["BB-IoU"]
            }
            data_dict[config_key].append(metrics_dict)

    # Adding normal bcos input for Energy and L1 loss to sparse and dilation data dicts
    bcos_input_energy = lossfn_data[("bcos", "BCos", "Input", "Energy")]
    bcos_input_l1 = lossfn_data[("bcos", "BCos", "Input", "L1")]
    dilation_data[("Energy", "0")] = bcos_input_energy
    dilation_data[("L1", "0")] = bcos_input_l1
    sparse_data[("Energy", "1")] = bcos_input_energy
    sparse_data[("L1", "1")] = bcos_input_l1

    return baseline_data, lossfn_data, dilation_data, sparse_data


def find_pareto_front(scores):
    """
    Identify the Pareto front points.
    Input: scores - A list of tuples, where each tuple represents a point in a 2D space.
    Output: A list of indices representing the points on the Pareto front.
    """
    # Convert list of tuples into a NumPy array for efficient calculations
    population = np.array(scores)
    population_size = len(population)
    pareto_front = np.ones(population_size, dtype=bool)

    for i in range(population_size):
        for j in range(population_size):
            if all(population[j] >= population[i]) and any(population[j] > population[i]):
                pareto_front[i] = 0
                break

    return population[pareto_front]


def sort_pareto_front(pareto_points):
    """
    Sort the Pareto front points in ascending order of F-Score.
    Input: pareto_points - A NumPy array of points on the Pareto front.
    Output: Sorted Pareto front points.
    """
    return pareto_points[np.argsort(pareto_points[:, 0])]


def find_limits(score, data_dict):
    """
    Find the limits for the x and y axes.
    Input: score - The score type to find the limits for.
    Output: The limits for the x and y axes as a tuple, with 10% padding.
    """
    scores = []
    for metrics_list in data_dict.values():
        scores.extend([metric[score] for metric in metrics_list])

    min_score = np.min(scores) * 100
    max_score = np.max(scores) * 100

    min_limit = min_score - min_score * 0.2
    max_limit = max_score + max_score * 0.1

    return min_limit, max_limit


def plot_domination(ax, baseline_x, baseline_y):
    """
    Plot the dominating and dominated areas.
    Input:
    - ax: The axis to plot on.
    - baseline_x: The x-coordinate of the baseline point.
    - baseline_y: The y-coordinate of the baseline point.
    """
    # Create green 'Dominating' area in the top-right quadrant of the baseline
    ax.fill_between([baseline_x, ax.get_xlim()[1]], baseline_y,
                    ax.get_ylim()[1], color='#B3E6B3', alpha=0.8, zorder=0)
    ax.plot([baseline_x, ax.get_xlim()[1]], [baseline_y, baseline_y], linestyle='--',
            color='#31852A', linewidth=1, zorder=0)  # Bottom line of dominating section
    ax.plot([baseline_x, baseline_x], [baseline_y, ax.get_ylim()[1]], linestyle='--',
            color='#31852A', linewidth=1, zorder=0)  # Left line of dominating section
    ax.text(ax.get_xlim()[1], baseline_y, 'Dominating', color='green',
            va='bottom', ha='right', fontsize=9, rotation=90)

    # Create grey 'Dominated' area in the bottom-left quadrant of the baseline
    ax.fill_between([ax.get_xlim()[0], baseline_x], ax.get_ylim()[
                    0], baseline_y, color='#E3E3E3', alpha=0.8, zorder=0)
    ax.plot([baseline_x, baseline_x], [ax.get_ylim()[0], baseline_y], linestyle='--',
            color='grey', linewidth=1, zorder=0)  # Right line of dominated section
    ax.plot([ax.get_xlim()[0], baseline_x], [baseline_y, baseline_y], linestyle='--',
            color='grey', linewidth=1, zorder=0)  # Top line of dominated section
    ax.text(ax.get_xlim()[0], ax.get_ylim()[0], 'Dominated',
            color='black', va='bottom', ha='left', fontsize=9)


def plot_baseline(ax, baseline_data, markers, colors, score_type):
    baseline_data = baseline_data[0]
    baseline_x, baseline_y = baseline_data["F-Score"] * \
        100, baseline_data[score_type] * 100
    ax.scatter(baseline_x, baseline_y, label="baseline",
               marker=markers["baseline"], color=colors["baseline"], edgecolors='black', s=69, zorder=3)
    plot_domination(ax, baseline_x, baseline_y)


def load_epg_star_json(root):
    baseline_data = {}
    lossfn_data = {}

    for model_file in os.listdir(root):
        filepath = os.path.join(root, model_file)
        with open(filepath, 'r') as json_file:
            json_data = json.load(json_file)

        parts = model_file.split("_")
        if "baseline" in model_file:
            data_dict = baseline_data
            model = parts[1]
            attr_method = parts[2]
            layer = parts[3]
            config_key = (model, attr_method, layer)
        else:
            data_dict = lossfn_data
            model = parts[0]
            attr_method = parts[1]
            layer = parts[2]
            loss_fn = parts[3]
            config_key = (model, attr_method, layer, loss_fn)

        # Initialise a list for this model if it doesn't exist yet
        if config_key not in data_dict:
            data_dict[config_key] = []

        # Add the metrics for this checkpoint to the list
        for checkpoint_name, metrics in json_data.items():
            metrics_dict = {
                "F-Score": metrics["F-Score"],
                "BB-Loc": metrics["BB-Loc"],
                "BB-IoU": metrics["BB-IoU"]
            }
            data_dict[config_key].append(metrics_dict)

    return baseline_data, lossfn_data


def plot_epg_star(plot_data, baseline_data, loss_fn1="Energy", loss_fn2="Energy*"):
    """
    Plot comparison of two loss functions.

    Input:
    - plot_data: Data for the Pareto front
    - baseline_data: Data for the baseline model
    - loss_fn1, loss_fn2: The two loss functions to compare
    """
    plt.rcParams['font.family'] = 'serif'
    markers = {"baseline": "X", loss_fn1: "o", loss_fn2: "*"}
    colours = {"baseline": "#FFFFFF", loss_fn1: "#ED3E63", loss_fn2: "#2E7AB5"}

    plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.gca()

    ax.set_xlim(75, 85)
    ax.set_ylim(50, 72)
    ax.set_ylabel("EPG Score (%)")
    ax.set_xlabel("F1 Score (%)")
    ax.set_xticks(np.arange(ax.get_xlim()[0] // 2 * 2, ax.get_xlim()[1], 2))
    ax.set_yticks(np.arange(ax.get_ylim()[0] // 5 * 5, ax.get_ylim()[1], 5))

    # Plot baseline data for comparison
    plot_baseline(ax, baseline_data[(
        'bcos', 'BCos', 'Input')], markers, colours, score_type="BB-Loc")

    # Plot data for both loss functions
    for loss_fn in [loss_fn1, loss_fn2]:
        config_key = ("bcos", "BCos", "Input", loss_fn)
        if config_key in plot_data:
            pareto_metrics = plot_data[config_key]
            pareto_points = np.array(
                [(model["F-Score"]*100, model["BB-Loc"]*100) for model in pareto_metrics])
            sorted_pareto_points = sort_pareto_front(pareto_points)
            if sorted_pareto_points.size > 0:
                f_scores, loc_scores = zip(*sorted_pareto_points)
                ax.scatter(
                    f_scores, loc_scores, label=f"{loss_fn}", marker=markers[loss_fn], color=colours[loss_fn], edgecolors='black', s=69, zorder=3)
                ax.plot(f_scores, loc_scores, linestyle='--',
                        color=colours[loss_fn], linewidth=2, zorder=1)

    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(
        0.5, 0.91), ncol=3, fontsize=9, frameon=True, framealpha=1)
    ax.set_title("Comparison of Energy and Energy* Loss Functions")

    plt.tight_layout()
    plt.show()


def load_seg_mask_json(root):
    baseline_data = {}
    lossfn_data = {}

    for model_file in os.listdir(root):
        filepath = os.path.join(root, model_file)
        with open(filepath, 'r') as json_file:
            json_data = json.load(json_file)

        parts = model_file.split("_")
        if "baseline" in model_file:
            data_dict = baseline_data
            model = parts[1]
            attr_method = parts[2]
            layer = parts[3]
            config_key = (model, attr_method, layer)
        else:
            data_dict = lossfn_data
            model = parts[0]
            attr_method = parts[1]
            layer = parts[2]
            config = parts[4].split(".")[0]
            config_key = (model, attr_method, layer, config)

        # Initialise a list for this model if it doesn't exist yet
        if config_key not in data_dict:
            data_dict[config_key] = []

        # Add the metrics for this checkpoint to the list
        for checkpoint_name, metrics in json_data.items():
            metrics_dict = {
                "F-Score": metrics["F-Score"],
                "BB-Loc": metrics["BB-Loc"],
                "BB-IoU": metrics["BB-IoU"]
            }
            data_dict[config_key].append(metrics_dict)

    return baseline_data, lossfn_data


def plot_seg_mask(plot_data, baseline_data):
    plt.rcParams['font.family'] = 'serif'
    markers = {"baseline": "X", "100-BB": "o", "9-BB": "v", "9-Seg": "p"}
    colours = {"baseline": "#FFFFFF", "100-BB": "#ED3E63",
               "9-BB": "#39D095", "9-Seg": "#FECB5B"}
    labels = {"100-BB": "100% BB", "9-BB": "9% BB", "9-Seg": "9% Seg"}

    plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.gca()

    ax.set_xlim(75, 85)
    ax.set_ylim(50, 75)
    ax.set_ylabel("EPG Score (%)")
    ax.set_xlabel("F1 Score (%)")
    ax.set_xticks(np.arange(ax.get_xlim()[0] // 2 * 2, ax.get_xlim()[1], 2))
    ax.set_yticks(np.arange(ax.get_ylim()[0] // 5 * 5, ax.get_ylim()[1], 5))

    # Plot baseline data for comparison
    plot_baseline(ax, baseline_data[(
        'bcos', 'BCos', 'Input')], markers, colours, score_type="BB-Loc")

    # Plot data for both loss functions
    for config in ["100-BB", "9-BB", "9-Seg"]:
        config_key = ("bcos", "BCos", "Input", config)
        if config_key in plot_data:
            pareto_metrics = plot_data[config_key]
            pareto_points = np.array(
                [(model["F-Score"]*100, model["BB-Loc"]*100) for model in pareto_metrics])
            pareto_points = find_pareto_front(pareto_points)
            sorted_pareto_points = sort_pareto_front(pareto_points)
            if sorted_pareto_points.size > 0:
                f_scores, loc_scores = zip(*sorted_pareto_points)
                ax.scatter(f_scores, loc_scores, label=labels[config], marker=markers[config],
                           color=colours[config], edgecolors='black', s=69, zorder=3)
                ax.plot(f_scores, loc_scores, linestyle='--',
                        color=colours[config], linewidth=2, zorder=1)

    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(
        0.5, 0.91), ncol=4, fontsize=9, frameon=True, framealpha=1)
    ax.set_title("Segmentation Mask & Sparse Annotations Results")

    plt.tight_layout()
    plt.show()


def load_seg_json(root):
    baseline_data = {}
    lossfn_data = {}

    for model_file in os.listdir(root):
        filepath = os.path.join(root, model_file)
        with open(filepath, 'r') as json_file:
            json_data = json.load(json_file)

        parts = model_file.split("_")
        if "baseline" in model_file:
            data_dict = baseline_data
            model = parts[1]
            attr_method = parts[2]
            layer = parts[3]
            config_key = (model, attr_method, layer)
        else:
            data_dict = lossfn_data
            config_key = parts[0].split(".")[0]

        # Initialise a list for this model if it doesn't exist yet
        if config_key not in data_dict:
            data_dict[config_key] = []

        # Add the metrics for this checkpoint to the list
        for _, metrics in json_data.items():
            metrics_dict = {
                "F-Score": metrics["F-Score"],
                "BB-Loc": metrics["BB-Loc"],
                "BB-IoU": metrics["BB-IoU"]
            }
            data_dict[config_key].append(metrics_dict)

    return baseline_data, lossfn_data


def plot_seg(plot_data, baseline_data):
    plt.rcParams['font.family'] = 'serif'
    markers = {"baseline": "X", "100% BB": "o", "8% BB": "v", "8% Seg": "p"}
    colours = {"baseline": "#FFFFFF", "100% BB": "#ED3E63",
               "8% BB": "#39D095", "8% Seg": "#FECB5B"}

    plt.figure(figsize=(6, 4), dpi=300)
    ax = plt.gca()

    ax.set_xlim(75, 85)
    ax.set_ylim(50, 78)
    ax.set_ylabel("EPG Score (%)", fontsize=12)
    ax.set_xlabel("F1 Score (%)", fontsize=12)
    ax.set_xticks(np.arange(ax.get_xlim()[0] // 2 * 2, ax.get_xlim()[1], 2))
    ax.set_yticks(np.arange(ax.get_ylim()[0] // 5 * 5, ax.get_ylim()[1], 5))

    # Plot baseline data for comparison
    plot_baseline(ax, baseline_data[(
        'bcos', 'BCos', 'Input')], markers, colours, score_type="BB-Loc")

    # Plot data for both loss functions
    for config in ["100% BB", "8% BB", "8% Seg"]:
        pareto_metrics = plot_data[config]
        pareto_points = np.array(
            [(model["F-Score"]*100, model["BB-Loc"]*100) for model in pareto_metrics])
        pareto_points = find_pareto_front(pareto_points)
        sorted_pareto_points = sort_pareto_front(pareto_points)
        if sorted_pareto_points.size > 0:
            f_scores, loc_scores = zip(*sorted_pareto_points)
            ax.scatter(f_scores, loc_scores, label=config, marker=markers[config],
                       color=colours[config], edgecolors='black', s=69, zorder=3)
            ax.plot(f_scores, loc_scores, linestyle='--',
                    color=colours[config], linewidth=2, zorder=1)

    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(
        0.5, 0.91), ncol=len(markers), fontsize=9, frameon=True, framealpha=1)
    ax.set_title("Segmentation Mask vs Bounding Box Annotations")

    plt.tight_layout()
    plt.show()
