import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict
from prettytable import PrettyTable

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# from toy_helper_2d import inertial_prox_naive, plot_contour, plot_gradient_flow, legend_without_duplicate_labels
from toy_helper_2d_es import inertial_prox_naive, plot_contour, plot_gradient_flow, legend_without_duplicate_labels #--> old
# from toy_helper_2d_aligator_hypergradient import inertial_prox_naive, plot_contour, plot_gradient_flow, legend_without_duplicate_labels # ---> May 2024


from toy_helper_adam_inertial import train_model_adam_adamom
from toy_other_opti_helper import train_model

# torch.autograd.set_detect_anomaly(mode=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def json_writer(config, name, path="results/"):
    with open(path+name+'.json', 'w') as fp:
        json.dump(config, fp, indent=4)

def main(test_function=None):
    if test_function == "beale":
        min_val, max_val = -4.5, 4.5
        lr, br, batch_size = 0.001, 0, 10
        plot_gradient_flow(min_val, max_val, lr, br, batch_size, file_name="2d_beale_gradient.png", func="beale")

    if test_function == "rosenbrock":
        min_val, max_val = -3, 5
        lr, br, batch_size = 0.0002, 0, 10 
        plot_gradient_flow(min_val, max_val, lr, br, batch_size, file_name="2d_rosenbrock_gradient.png", func="rosenbrock")
        # mapper = naive_morse_smale_complex(min_val, max_val, lr=0.0002, br=0, func="rosenbrock")
        # morse_complex_plot(x, y, min_val, max_val, mapper, filename = "2d_rosenbrock.png", precision_pt = 5, func="rosenbrock")

    if test_function == "goldstein":
        min_val, max_val = -4, 4
        lr, br, batch_size = 0.0001, 0, 10 
        plot_contour(min_val, max_val, "2d_goldstein_gradient.png", "goldstein-price")
        plot_gradient_flow(min_val, max_val, lr, br, batch_size, file_name="2d_goldstein_gradient.png", func="goldstein-price")

    if test_function == "ackley":
        min_val, max_val = -4, 4
        lr, br, batch_size = 0.01, 0, 10 
        plot_contour(min_val, max_val, "2d_ackley.png", "ackley")
        plot_gradient_flow(min_val, max_val, lr, br, batch_size, file_name="2d_ackley_gradient.png", func="ackley")


def plot_trajectory(point_tuple, history, gradient_list, cost, plt, colors, ax, func=None, marker_style_map=None, optimizer="sgd"):
        """ plot for one point """
        # plot style details
        # color = ["red", "pink", "orange"]

        xcoord, ycoord = point_tuple
        print("*"*89)
        print("point taken ....", xcoord, ycoord)

        loop_iter = 0
        print("No. of gradient points:", len(history))
        for (i, (beta_i, grad_i)) in enumerate(zip(history, gradient_list)):
            if i == 0:
                plt.scatter([beta_i[0]], [beta_i[1]], s=40, c='black', marker="o", linewidth=5.0, label=str(loop_iter))

            if i%1 == 0:
                print("Epoch:", i, "beta_0: ",[beta_i[0]], "beta_1: " , [beta_i[1]], "beta norm: ", np.linalg.norm(beta_i), "grad norm: ", np.linalg.norm(grad_i), "cost: ", cost[i])
                if optimizer in ['sgd', 'adam']:
                    if i%50 == 0:
                        plt.scatter([beta_i[0]], [beta_i[1]], s=20*4, c=colors, marker=marker_style_map, linewidth=1, label=str(loop_iter)) #mediumpurple
                else:
                    # if i > 40:
                    if i%50 == 0:
                        plt.scatter([beta_i[0]], [beta_i[1]], s=40+20, c=colors, marker=marker_style_map, linewidth=1, label=str(loop_iter)) #mediumpurple
                    else:
                        plt.scatter([beta_i[0]], [beta_i[1]], s=40+20, c=colors, marker=marker_style_map, linewidth=1, label=str(loop_iter)) #mediumpurple
                

            if i == len(history)-1: # last point with black color
                print(i)
                if optimizer not in ['sgd', 'adam']:
                    plt.scatter([beta_i[0]], [beta_i[1]], s=70, c=colors, marker="d", linewidth=2.0, label=str(loop_iter))
                    loop_iter+=1
        
        # for inertial_sgd, plot 1
        if func == "beale":
            plt.plot(3,0.5, c='black', marker="*", markersize=30) #markersize=18
            # plt.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c="black", linewidth=1)
            if optimizer == 'sgd':
                plt.plot(-0.13977523148059845,1.8211967945098877, c='limegreen', marker="d", markersize=10+5) # final point for adam
            elif optimizer == 'adam':
                plt.plot(-0.19078427648544312, 1.8211575746536255, c='royalblue', marker="d", markersize=10+5) # final point for gd

        # if func == "beale":
        #     plt.plot(3,0.5, c='black', marker="*", markersize=18)
        #     # plt.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c="black", linewidth=1)
        #     if optimizer == 'sgd':
        #         plt.plot(-0.13977575302124023,1.8211945295333862, c='green', marker="d", markersize=10) # final point for adam
        #     elif optimizer == 'adam':
        #         plt.plot(-0.1600834023952484, 1.8213022947311401, c='blue', marker="d", markersize=10)
                         
        if func == "ackley":
            plt.plot(0,0, 'k*', markersize=18)
            plt.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c=colors, linewidth=1)

        legend_without_duplicate_labels(ax)
        marker_red = Line2D([0], [0], label='GD + MoXCo', marker="*", markersize=15, markeredgecolor='r', markerfacecolor='r', linestyle='--', color='red')
        # marker_red = Line2D([0], [0], label='Adam + MoXCo', marker="*", markersize=15, markeredgecolor='r', markerfacecolor='r', linestyle='--', color='red')
        marker_blue = Line2D([0], [0], label='Adam', marker=">", markersize=15, markeredgecolor='royalblue', markerfacecolor='royalblue', linestyle='--', color='royalblue')
        marker_green = Line2D([0], [0], label='GD', marker=".", markersize=15, markeredgecolor='limegreen', markerfacecolor='limegreen', linestyle='--', color='limegreen')
        plt.legend(handles=[marker_blue, marker_green, marker_red], fontsize=20)

def compare_against_others(point_tuple, func="beale"):
    ''' function that calls 2d train for vanilla ADAM and others to plot comparative figure'''

    optimizers = ['gd+inertial', 'adam', 'sgd']
    # optimizers = ['adam+inertial','adam', 'sgd']

    map_list = {"beale":[-4,4], "ackley":[-3,3.5]}
    print(map_list[func] )
    min_val, max_val = map_list[func] 
    file_name = f"comparison_plot_2024_{func}.png" #new -->2024

    X_grid, Y_grid, filename, ax = plot_contour(min_val, max_val, file_name, func=func)
    xcoord, ycoord = point_tuple
    config = {}

    def optim_output(optim):
        if optim == "gd+inertial":
            # plot 1/plot 2
            # br, batch_size, offset, epochs, warm_start, eta, alpha, beta = 0, 10, 0.001, 1400, True, 0.80, 0.99, 0.99
            # ackley
            # br, batch_size, offset, epochs, warm_start, eta, alpha, beta = 0, 10, 0.044, 1400, True, 0.89, 0.9, 0.94
            # br, batch_size, offset, epochs, warm_start, eta, alpha, beta = 0, 10, 0.05, 1400, True, 0.88, 0.9, 0.94


            # latest after correction
            br, batch_size, offset, epochs, warm_start, eta, alpha, beta = 0, 10, 0.005, 800, True, 0.86, 0.8, 0.8
            config[optim] = {"epochs":epochs, "warm_start":warm_start, "restart window":eta, "alpha":alpha, "beta":beta}
            history, _, cost, gradient_list, _ = inertial_prox_naive([xcoord, ycoord], alpha, beta, epochs, offset, test_func=func, warm_start=warm_start, eta=eta)
            return history, cost, gradient_list

        if optim == "adam":
            lr , epochs = 0.01, 800#0.001, 800 # plot 1 | # lr, epochs = 0.001, 1400 # plot 2
            # lr , epochs = 0.05, 1400
            config[optim] = {'lr':lr, "epochs":epochs}
            history, list_gradient, cost = train_model(point_tuple, epochs, lr, optimizer=optim, test_func=func)
            return history, cost, list_gradient
        
        if optim == "sgd":
            lr , epochs = 0.01, 800#0.001, 800 # plot 1 |  # lr, epochs = 0.001,1400 plot 2
            # lr , epochs = 0.05, 1400
            config[optim] = {'lr':lr, "epochs":epochs}
            history, list_gradient, cost = train_model(point_tuple, epochs, lr, optimizer=optim, test_func=func)
            return history, cost, list_gradient
        
        if optim == "adam+inertial":
            # alpha, beta, initial_guess, lr, epochs, temperature = 0.99, 0.99, point_tuple, 0.001, 800, 0.01 # plot 2
            # alpha, beta, initial_guess, lr, epochs, temperature = 0.9, 0.9, point_tuple, 0.01, 800, 0.01

            alpha, beta, initial_guess, lr, epochs, temperature = 0.96, 0.4, point_tuple, 0.005, 800, 0.15
            config[optim] = {'lr':lr, "epochs":epochs, "alpha":alpha, "beta":beta, "temperature":temperature}
            history, list_gradient, cost = train_model_adam_adamom(alpha, beta, initial_guess, lr, nb_epochs=epochs, func="beale", temperature=temperature)
            return history, cost, list_gradient

    # colormap = ['red', 'blue', 'green']
    colormap = ['red', 'royalblue', 'limegreen']
    linestyles = ["dashed", ":", "-.", "-", "--"]
    marker_styles = ["*", ">", "h", "o"]
    config['init_point'] = point_tuple
    cost_list = {}
    for i, optim in enumerate(optimizers):
        print("*"*20, f"Running optimizer: {optim}", "*"*20)
        history, cost, gradient_list = optim_output(optim)
        cost_list[optim] = cost
        
        if history !=0 and cost!=0 and gradient_list!=0:
            plot_trajectory(point_tuple, history, gradient_list, cost, plt, colormap[i], ax, func=func, marker_style_map=marker_styles[i], optimizer=optim)

    json_writer(config,f"comparison_plot_2024_{func}") # save config, new -->2024
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    ########################
    # plt.style.use('seaborn-darkgrid')
    plt.title(f"Init point: {point_tuple}")
    optim_names = {"gd+inertial":"GD + MoXCo","adam":"Adam","sgd":"GD"}
    # optim_names = {"adam+inertial":"Adam + MoXCo","adam":"Adam","sgd":"GD"}
    count = 0
    for optim, cost in cost_list.items():
        plt.plot(list(range(len(cost))), cost, label=f"{optim_names[optim]}", color=colormap[count], linestyle="--")
        plt.xlabel("iterations")
        plt.ylabel("Training loss")
        plt.legend(loc="upper right")
        plt.ylim([-1, max(cost)])
        count+=1
    plt.legend(fontsize=20)
    plt.savefig("loss.png")
    plt.close()


# main(test_function="beale")
# plot 1
# compare_against_others((-3.5,1.5), func="beale")

# plot 1 & plot 2
compare_against_others((-2.5,1.2), func="beale")


# compare_against_others((0,-2.5), func="beale")
# compare_against_others((-2,3), func="ackley")
# compare_against_others((-2,1.3), func="beale")
# compare_against_others((-2,3), func="ackley")