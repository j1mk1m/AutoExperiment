import json
import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

from argparse import ArgumentParser

custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
sns.set_theme()
sns.set(font_scale=2) 


# args.group = 'Occupation'

def parser_args():
    parser = ArgumentParser(description="JMEF Postprocessing")

    parser.add_argument(
        '--figures',
        type=int,
        nargs='+',
        default=[2,3,4],
        help="Figures to construct"
    )

    parser.add_argument(
        '--fig2_models', 
        type=str, 
        nargs='+', 
        default=['BPRMF'],
        help="Model(s) to construct figure 2"
    )

    parser.add_argument(
        '--fig3_models',
        type=str,
        nargs='+',
        default=['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF'],
        help="Model(s) to construct figure 3"
    )

    parser.add_argument(
        '--fig4_models', 
        type=str,
        nargs='+', 
        default=['all'], 
        help="Model(s) to construct figure 4"
    )

    parser.add_argument(
        '--fig5_models',
        type=str,
        nargs='+',
        default=['Bert4Rec'],
        help="Model(s) to construct figure 5"
    )


    
    parser.add_argument(
        '--group',
        type=str,
        default='Gender',
        help="User-side group"
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./src/outputs/figures/',
        help='Path to output directory'
    )

    parser.add_argument(
        '--original_results',
        type=str,
        default='Y',
        help='Plot our original reproduction results or private rerun'
    )

    return parser.parse_args()



def load_data(model, Experiment_nr, group, metric_list, apply_min_max = False, max_ = 0, min_ = 0, fig4 = False):
    """ Load experiments for plotting
    """
    if args.original_results ==  'Y':
        result_dir = 'original_experiments/'
    else:
        result_dir = ''
    df = pd.DataFrame(columns = column_names)
    df["II"] = [[], [], []]
    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            if group == 'Age':
                metrics = json.load(open('./src/outputs/ml-1m/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '_Y.json', 'r'))
                static  = json.load(open('./src/outputs/ml-1m/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static_Y.json', 'r'))
            elif group == 'Occupation':
                metrics = json.load(open('./src/outputs/ml-1m/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('./src/outputs/ml-1m/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            elif group == 'Gender':
                metrics = json.load(open('./src/outputs/ml-1m/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('./src/outputs/ml-1m/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + group +'/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            elif group == 'lt':
                metrics = json.load(open('./src/outputs/lt/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + metric_list[metric][component]+ '_all_' + model + '.json', 'r'))
                static  = json.load(open('./src/outputs/lt/' + result_dir + 'Experiment_' + str(Experiment_nr) + '_' + model + '/' + metric_list[metric][component] +'_all_' + model + '_static.json', 'r'))
            else:
                print('Unknown group')

            if fig4:
                df[column_names[metric]][component] = metrics
            else:
                metrics.extend(static)

            if apply_min_max == True:
                try:
                    metrics = min_max(np.array(metrics), max_ = max_[component][metric], min_ = min_[component][metric])
                except TypeError:
                    metrics = min_max(np.array(metrics), max_ = np.max(metrics), min_ = np.min(metrics))
            else:
                metrics = np.array(metrics)
                
            df[column_names[metric]][component] = metrics
    
    df.index = ['F', 'D', 'R']
    return df


def AUC_trap(x, y, minval):
    """ Trapezoidal Rule for calculating Area under curve
    Input:
        x: x-data
        y: y-data
        minval: set value for which to integrate upto
    
    Output:
        Area: area under curve
    """
    AUC = 0
    minval = 1
    for i in range(len(x)-2):
        if x[i+1] <= minval:
            AUC += 0.5*(y[i+1]+ y[i])*(x[i+1] - x[i])
    return AUC 

def find_global_min_max(models):
    global_min = np.ones((3, 6))*100
    global_max = np.zeros((3, 6))
    for model in models.values():
        for i in range(len(model)):
            for j in range(len(model.iloc[i])):
                if global_min[i][j] > np.min(model.iloc[i][j]):
                    global_min[i][j] = np.min(model.iloc[i][j])
                if global_max[i][j] < np.max(model.iloc[i][j]):
                    global_max[i][j] = np.max(model.iloc[i][j])
    return global_min, global_max


def min_max(arr, max_ = 0, min_ = 0):
    if max_ == 0:
        max_ = np.max(arr)
    if min == 0:
        min_ = np.min(arr)
    min_max = (arr-min_)/(max_-min_)
    return min_max



def figure_2(data, name, column_names):
    """ Reproduce figure 2
    """
    fig, axs = plt.subplots(3, 6, constrained_layout=True, figsize = (20, 10))
    beta_values = ['8', '4', '2', '1', '1/2', '1/4','1/8', 'ST']
    
    x_axis = np.arange(len(beta_values))

    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            metrics = data[column_names[metric]][component]
            sns.barplot(ax=axs[component, metric], x=x_axis - 0.2, y=metrics, palette='Blues_d')
            axs[component, metric].plot(x_axis - 0.2, metrics, '--.', color = 'r')
            axs[component, metric].set_xticks(x_axis, beta_values, rotation = 60)
            axs[component, metric].set_title(metric_list[metric][component])
    
    for ax in fig.get_axes():
       ax.label_outer()  

    fig.supxlabel(r'$\beta$')

    plt.savefig('{}Figure_2_{}_{}.png'.format(args.output_dir, name, args.group), bbox_inches='tight')
    print("Figure 2 saved to {}".format(args.output_dir))
    plt.clf()


def figure_3(models, model_name, column_names, group):
    """ Reproduce Figure 3 - Disparity-relevance tradeoff curves
    """
    fig, axs = plt.subplots(1, 6, constrained_layout=True, figsize = (20, 3.5))
    
    x_axis = np.arange(len(x_values))
    x_min_vals = np.ones(len(column_names))
    for model in models:
        for metric in range(len(column_names)):
                x_min_vals[metric] = model[column_names[metric]][1][-1]

    for i in range(len(models)):
        model = models[i]
        AUC_list = []
        for metric in range(len(column_names)):
            axs[metric].plot(model[column_names[metric]][1], model[column_names[metric]][2], '--*', label = model_name[i])
            axs[metric].set_xlabel(column_names[metric] + '-D') 
            axs[metric].set_ylabel(column_names[metric] + '-R')
            AUC_list.append(np.round(AUC_trap(model[column_names[metric]][1], model[column_names[metric]][2], x_min_vals[metric]), 5))

        # Print AUC table
        print('AUC Table:')
        print(model_name[i], ' & ' , AUC_list[0], ' & ', AUC_list[1], ' & ', AUC_list[2], ' & ', AUC_list[3], ' & ', AUC_list[4], ' & ',AUC_list[5], ' \\\\ ')
    fig.legend(labels=model_name, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=5)



    plt.savefig('{}Figure_3_{}.png'.format(args.output_dir, args.group), bbox_inches = 'tight')
    print("Figure 3 saved to {}".format(args.output_dir))
    plt.clf()


def figure_4(models, group, column_names):
    """ Reproduce Figure 4 - Kendall rank correlation heatmap
    """
    if group == 'Age':
        F = relative_difference(age_F, age_F_new)
        R = relative_difference(age_R, age_R_new)
        D = relative_difference(age_D, age_D_new)
    elif group == 'Gender':
        F = relative_difference(gender_F, gender_F_new)
        R = relative_difference(gender_R, gender_R_new)
        D = relative_difference(gender_D, gender_D_new)
    
    rel_diff = [F, D, R]

    df = pd.DataFrame()
    for model in models:
        df = pd.concat((df, model), axis=0, join='outer')

    for column in column_names:
        df = df.explode(column_names)

    main_metric = df.loc['F'].astype(float)
    disparity = df.loc['D'].astype(float)
    relevance = df.loc['R'].astype(float)

    main_corr = main_metric.corr(method='kendall')
    d_corr = disparity.corr(method='kendall')
    r_corr = relevance.corr(method='kendall')
    titles = [group + ' - F', group + ' - D', group + ' - R']
    j = 0
    for i in [main_corr, d_corr, r_corr]:
        sns.set(font_scale=1.3)
        plot= sns.heatmap(i, cbar=False, annot=True, cmap="YlGnBu", fmt='.3g')
        #Set all sides
        plt.text(6.1, 7, str(np.round(rel_diff[j][0],2))+'\n\n\n' + str(np.round(rel_diff[j][1],2))+'\n\n\n' + str(np.round(rel_diff[j][2],2))+'\n\n\n' +str(np.round(rel_diff[j][3],2))+'\n\n\n' + str(np.round(rel_diff[j][4],2))+'\n\n\n' + str(np.round(rel_diff[j][5],2))+'\n\n\n', horizontalalignment='left', size= 14.5, color='black', weight = 'bold')
        plt.text(6.67, -0.25, "ARD", horizontalalignment='right', size= 14.5, weight = 'bold', color='black')
        plt.title(titles[j])
        plt.savefig('{}Figure_4_{}_{}'.format(args.output_dir, group, j), bbox_inches='tight')
        
        plt.clf()

        j += 1
    print("Figure 4 saved to {}".format(args.output_dir))


def figure_5(data_set, column_names):
    """ Constructing figure 5
    """
    fig, axs = plt.subplots(3, 6, constrained_layout=True, figsize = (20, 10))
    beta_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']
    
    x_axis = np.arange(len(beta_values))

    d1, d2 = data_set.items()
    

    
    
    for metric in range(len(metric_list)):
        for component in range(len(metric_list[metric])):
            
            metrics_d1 = d1[1][column_names[metric]][component]
            metrics_d2 = d2[1][column_names[metric]][component]
            dict_d1 = {'8': metrics_d1[0], '4': metrics_d1[1], '2': metrics_d1[2], '1': metrics_d1[3], '1/2': metrics_d1[4], '1/4': metrics_d1[5], '1/8': metrics_d1[6], 'ST': metrics_d1[7]} 
            dict_d2 = {'8': metrics_d2[0], '4': metrics_d2[1], '2': metrics_d2[2], '1': metrics_d2[3], '1/2': metrics_d2[4], '1/4': metrics_d2[5], '1/8': 0, 'ST': metrics_d2[6]} 
            df1 = pd.DataFrame.from_dict(dict_d1, orient='index', columns=[d1[0]])
            df1['key'] = d1[0]
            df1['beta'] = x_axis
            df2 = pd.DataFrame.from_dict(dict_d2, orient='index', columns=[d1[0]])
            df2['key'] = d2[0]
            df2['beta'] = x_axis
            df = pd.concat([df1, df2])
            
            sns.barplot(ax=axs[component, metric], x="beta", y = d1[0], hue = "key", data = df, palette=['tab:orange', 'lightblue']) 
            #sns.barplot(ax=axs[component, metric], x=x_axis - 0.2, y=metrics, palette='Blues_d')
            #axs[component, metric].plot(x_axis - 0.2, metrics, '--.', color = 'r')
            axs[component, metric].set_xticks(x_axis, beta_values, rotation = 60)
            #axs[component, metric].set_xlabel(r'$\beta$')
            axs[component, metric].set_ylabel('')
            axs[component, metric].set_xlabel('')
            axs[component, metric].set_title(metric_list[metric][component])
            axs[component, metric].get_legend().remove()
    
    for ax in fig.get_axes():
       ax.label_outer()  
    # remove the x and y ticks
    fig.legend(labels=['ml-1m', 'LT'], loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    
    fig.supxlabel(r'$\beta$')
    plt.savefig('{}Figure_5_{}_{}.png'.format(args.output_dir, d1[0] + d2[0], args.group), bbox_inches='tight')
    print("Figure 5 saved to {}".format(args.output_dir))
    plt.clf()
    

if __name__ == '__main__':
    args = parser_args()

    metric_list = [['IIF','IID','IIR'], 
                   ['IGF','IGD','IGR'],  
                   ['GIF','GID','GIR'],  
                   ['GGF','GGD','GGR'],  
                   ['AIF','AID','AIR'],  
                   ['AGF','AGD','AGR']]

    column_names = ["II", "IG", "GI", "GG", "AI", "AG"]
    x_values = ['8', '4', '2', '1', '1/2', '1/4', '1/8', 'ST']

    # Experiment number of corresponding models
    exp_nr = {
        'BPRMF': 1, 'LDA': 2, 'PureSVD': 3,'SLIM': 4,
        'WRMF': 5, 'CHI2': 6, 'HT': 7, 'KLD': 8,
        'LMWI': 9, 'LMWU': 10, 'SVD': 11, 'NNI': 12,
        'NNU': 13, 'PLSA': 14, 'Random': 15, 'RM1': 16,
        'RM2': 17, 'RSV': 18, 'RW': 19, 'UIR': 20,
        'Bert4Rec': 21
    }
    # Numerical results
    age_F = np.array([[1.000 , 0.487 , 0.432 , 0.307 , 0.444 , 0.327],
            [0.487 , 1.000 , 0.611 , 0.576 , 0.568 , 0.551],
            [0.432 , 0.611 , 1.000 , 0.723 , 0.949 , 0.745],
            [0.307 , 0.576 , 0.723 , 1.000 , 0.683 , 0.911],
            [0.444 , 0.568 , 0.949 , 0.683 , 1.000 , 0.706],
            [0.327 , 0.551 , 0.745 , 0.911 , 0.706 , 1.000]])
    age_R = np.array([[1.000 , 0.623 ,0.283 ,0.318 ,0.266   , 0.232 ],
            [0.623 , 1.000 ,-0.077, -0.035, -0.095, -0.123],
            [0.283 , -0.077, 1.000, 0.870 , 0.980 , 0.868 ],
            [0.318 , -0.035, 0.870, 1.000 , 0.857 , 0.905 ],
            [0.266 , -0.095, 0.980, 0.857 , 1.000 , 0.869 ],
            [0.232 , -0.123, 0.868, 0.905 , 0.869 , 1.000 ]])
    age_D = np.array([[1.000 , 0.441 ,0.344 ,0.321 ,0.349,  0.318     ],
            [0.441 , 1.000 ,-0.030, -0.056, -0.028 , -0.058 ],
            [0.344 , -0.030, 1.000, 0.900 , 0.991 , 0.893 ],
            [0.321 , -0.056, 0.900, 1.000 , 0.900 , 0.976 ],
            [0.349 , -0.028, 0.991, 0.900 , 1.000 , 0.889 ],
            [0.318 , -0.058, 0.893, 0.976 , 0.889 , 1.000]])
    gender_F = np.array([
                [1.000 , 0.483 , 0.504 , 0.363 , 0.446 , 0.323],
                [0.483 , 1.000 , 0.696 , 0.617 , 0.566 , 0.544],
                [0.504 , 0.696 , 1.000 , 0.699 , 0.791 , 0.659],
                [0.363 , 0.617 , 0.699 , 1.000 , 0.639 , 0.786],
                [0.446 , 0.566 , 0.791 , 0.639 , 1.000 , 0.702],
                [0.323 , 0.544 , 0.659 , 0.786 , 0.702 , 1.000]])
    gender_R = np.array([[1.000 , 0.620 , 0.320 , 0.763 , 0.273 , 0.239 ],
                [0.620 , 1.000 , -0.042, 0.490 , -0.097, -0.120],
                [0.320 , -0.042, 1.000 , 0.384 , 0.911 , 0.868 ],
                [0.763 , 0.490 , 0.384 , 1.000 , 0.365 , 0.315 ],
                [0.273 , -0.097, 0.911 , 0.365 , 1.000 , 0.865 ],
                [0.239 , -0.120, 0.868 , 0.315 , 0.865 , 1.000 ]])
    gender_D = np.array([[1.000 , 0.520  , 0.334 , 0.450 , 0.303 , 0.279 ],
                [0.520 , 1.000  , -0.003, 0.127 , -0.032, -0.060],
                [0.334 , -0.003 , 1.000 , 0.785 , 0.912 , 0.862 ],
                [0.450 , 0.127  , 0.785 , 1.000 , 0.772 , 0.739 ],
                [0.303 , -0.032 , 0.912 , 0.772 , 1.000 , 0.882 ],
                [0.279 , -0.060 , 0.862 , 0.739 , 0.882 , 1.000 ]])
    gender_F_new = np.array([
                [1.000000,  0.460021,  0.411305,  0.331963,  0.428160,  0.353546],
                [0.460021,  1.000000,  0.667215,  0.614183,  0.630216,  0.583556],
                [0.411305,  0.667215,  1.000000,  0.747174,  0.953546,  0.746146],
                [0.331963,  0.614183,  0.747174,  1.000000,  0.713875,  0.921686],
                [0.428160,  0.630216,  0.953546,  0.713875,  1.000000,  0.714491],
                [0.353546,  0.583556,  0.746146,  0.921686,  0.714491,  1.000000]
                ])
    gender_R_new = np.array([
                [1.000000,  0.675642,  0.589517,  0.573279,  0.580267,  0.496197],
                [0.675642,  1.000000,  0.278726,  0.290853,  0.268654,  0.213361],
                [0.589517,  0.278726,  1.000000,  0.859198,  0.982939,  0.851182],
                [0.573279,  0.290853,  0.859198,  1.000000,  0.848715,  0.914286],
                [0.580267,  0.268654,  0.982939,  0.848715,  1.000000,  0.848099],
                [0.496197,  0.213361,  0.851182,  0.914286,  0.848099,  1.000000]
    ])
    gender_D_new = np.array([
        [1.000000,  0.359918,  0.213361,  0.150874,  0.216033,  0.155396],
        [0.359918,  1.000000,  0.030832,  0.055087,  0.035149,  0.051799],
        [0.213361,  0.030832,  1.000000,  0.824049,  0.989517,  0.825283],
        [0.150874,  0.055087,  0.824049,  1.000000,  0.823022,  0.965468],
        [0.216033,  0.035149,  0.989517,  0.823022,  1.000000,  0.823022],
        [0.155396,  0.051799,  0.825283,  0.965468,  0.823022,  1.000000]])
    age_F_new = np.array([
        [1     , 0.461 , 0.431 , 0.365 , 0.421 , 0.354],
        [0.461 , 1     , 0.668 , 0.63  , 0.63  , 0.583],
        [0.431 , 0.668 , 1     , 0.709 , 0.939 , 0.723],
        [0.365 , 0.63  , 0.709 , 1     , 0.696 , 0.924],
        [0.421 , 0.63  , 0.939 , 0.696 , 1     , 0.713],
        [0.354 , 0.583 , 0.723 , 0.924 , 0.713 , 1]])
    age_R_new = np.array([
        [1.000000,  0.671737,  0.594861,  0.676259,  0.579239,  0.499075],
        [0.671737,  1.000000,  0.284687,  0.408839,  0.271942,  0.215622],
        [0.594861,  0.284687,  1.000000,  0.818705,  0.974923,  0.858582],
        [0.676259,  0.408839,  0.818705,  1.000000,  0.801028,  0.785406],
        [0.579239,  0.271942,  0.974923,  0.801028,  1.000000,  0.850771],
        [0.499075,  0.215622,  0.858582,  0.785406,  0.850771,  1.000000]])
    age_D_new = np.array([[1.000000,  0.309147,  0.193011,  0.167934,  0.202672,  0.131963],
        [0.309147,  1.000000,  0.038232,  0.086331,  0.033094,  0.050360],
        [0.193011,  0.038232,  1.000000,  0.834738,  0.966906,  0.838232],
        [0.167934,  0.086331,  0.834738,  1.000000,  0.823022,  0.893320],
        [0.202672,  0.033094,  0.966906,  0.823022,  1.000000,  0.826516],
        [0.131963,  0.050360,  0.838232,  0.893320,  0.826516,  1.000000]])

    def relative_difference(x,y):
        diff = np.mean(np.abs((x-y)/x), axis = 1)
        
        return diff


    if 2 in args.figures:
        print("\nReconstruct figure 2 on group: {}".format(args.group))
        print("Models: {}".format(args.fig2_models))
        # Load unnormalized data
        all_models = {}
        for model_name in args.fig2_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list)
            all_models[model_name] = model_data

        # Find global min and max across all models per metric per component
        global_min, global_max = find_global_min_max(all_models)

        all_models_norm = {}
        for model_name in args.fig2_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list, apply_min_max=True, max_ = global_max, min_ = global_min)
            all_models_norm[model_name] = model_data
    
        for name, data in all_models_norm.items():
            figure_2(data, name, column_names)

        
    if 3 in args.figures:
        print("\nReconstruct figure 3 on group: {}".format(args.group))
        print("Models: {}".format(args.fig3_models))
        all_models = {}
        for model_name in args.fig3_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list)
            all_models[model_name] = model_data

        # Find global min and max across all models per metric per component
        global_min, global_max = find_global_min_max(all_models)

        all_models_norm = {}
        for model_name in args.fig3_models:
            model_data = load_data(model_name, exp_nr[model_name], args.group, metric_list, apply_min_max=True, max_ = global_max, min_ = global_min)
            all_models_norm[model_name] = model_data
    
        figure_3(list(all_models_norm.values()), list(all_models_norm.keys()), column_names, args.group)

            # Numerical experimental values
        original_results = np.array([[0.6331, 0.4774, 0.2904, 0.2953, 0.2712, 0.2814],
                                [0.5664, 0.4088, 0.2837, 0.3164, 0.2687, 0.3115],
                                [0.5830, 0.4102, 0.2921, 0.3030, 0.2755, 0.2942],
                                [0.6408, 0.4654, 0.2776, 0.2851, 0.2605, 0.2752],
                                [0.5996, 0.4769, 0.3135, 0.3186, 0.2957, 0.3139]])

        our_results = np.array([[0.6336, 0.4370, 0.1949, 0.2487, 0.1879, 0.0869],
                            [0.5567, 0.2448, 0.2305, 0.1877, 0.2007, 0.1799],
                            [0.5691, 0.3239, 0.2615, 0.2896, 0.0712, 0.0909],
                            [0.6540, 0.3757, 0.2591, 0.2739, 0.0799, 0.1145],
                            [0.5901, 0.5171, 0.2826, 0.3340, 0.2194, 0.2219]])
        """
        # Results when estimating without minimum treshhold 
        our_results = np.array([[0.6335, 0.53775, 0.35547,  0.37903,  0.33242,  0.3051],
                                [0.5567, 0.50902, 0.46713,  0.18772,  0.41484,  0.1799],
                                [0.5691, 0.32391, 0.57335,  0.50482,  0.49946,  0.3852],
                                [0.6540, 0.51285, 0.61406,  0.53381,  0.61414,  0.5383],
                                [0.5901, 0.51706, 0.28261,  0.33397,  0.21939,  0.2219]])
        """
        avg_rel_diff = []
        
        for i in range(5):
            temp = np.mean(np.abs((our_results[i] - original_results[i])/(original_results[i])))
            print(np.round(temp,2))    
            avg_rel_diff.append(temp)
        print('Avg. avg. difference = ', np.round(np.mean(avg_rel_diff), 2), '+-', np.round(np.std(avg_rel_diff), 2))



    if 4 in args.figures:
        print("\nReconstruct figure 4 on group: {}".format(args.group))
        
        if 'all' in args.fig4_models:
            mod_names = ['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF', 'CHI2', 'HT', 'KLD', 'LMWI', 'LMWU', 'SVD', 'NNI', 'NNU', 'PLSA', 'Random', 'RM1', 'RM2', 'RSV', 'RW', 'UIR']
            #mod_names = ['BPRMF', 'LDA', 'PureSVD', 'SLIM', 'WRMF', 'CHI2', 'HT', 'KLD',  'SVD', 'PLSA', 'Random', 'RM1', 'RM2', 'RSV', 'RW', 'UIR'] 
            print("Models: {}".format(args.fig4_models))
        else:
            mod_names = args.fig4_models
            print(mod_names)

        model_data = []
        for model_name in mod_names:
            model_load = load_data(model_name, exp_nr[model_name], args.group, metric_list, fig4=True)
            model_data.append(model_load)

        figure_4(model_data, args.group, column_names)

    if 5 in args.figures:
        print("\nReconstruct figure 5 on group: {}".format(args.group))
        print("Models: {}".format(args.fig5_models))
        # Load unnormalized data
        all_models = {}
        model_data = load_data(args.fig5_models[0], exp_nr[args.fig5_models[0]], 'Age', metric_list, apply_min_max=True)    
        all_models[args.fig5_models[0] + 'ML'] = model_data
        model_data = load_data(args.fig5_models[0], exp_nr[args.fig5_models[0]], 'lt', metric_list, apply_min_max=True)    
        all_models[args.fig5_models[0] +'LT'] = model_data
        
        figure_5(all_models, column_names)




                        


