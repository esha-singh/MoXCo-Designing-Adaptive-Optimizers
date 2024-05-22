#comparison code for adam, adamw trajectory any plots we might need for 2D examples.
# related to review notes heavytails workshop

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from toy_helper_2d_aligator_test import xy_meshgrid, objective_function_plot_xy_independent, inertial_prox_naive
# from toy_helper_2d_Mox_adam import xy_meshgrid, objective_function_plot_xy_independent, inertial_prox_naive
from toy_helper_2d_aligator_hypergradient import xy_meshgrid, objective_function_plot_xy_independent, inertial_prox_naive

# from toy_helper_adam_inertial import train_model_adam_adamom
# from toy_other_opti_helper import train_model
plot_scale = 1.25
plt.rcParams["figure.figsize"] = (plot_scale*16, plot_scale*9)


def compare_against_others(point_tuple, optimizers, func="beale"):
    ''' function that calls 2d train for vanilla ADAM and others to plot comparative figure'''

    xcoord, ycoord = point_tuple
    res = []
    def optim_output(optim):
        if optim == "adam":
            # lr , epochs = 0.001, 800 # plot 1 | # lr, epochs = 0.001, 1400 # plot 2
            lr , epochs = 0.05, 1400
            history, list_gradient, cost = train_model(point_tuple, epochs, lr, optimizer=optim, test_func=func)
            return history, cost, list_gradient
        
        if optim == "adamW":
            # lr , epochs = 0.001, 800 # plot 1 |  # lr, epochs = 0.001,1400 plot 2
            lr , epochs = 0.05, 1400
            history, list_gradient, cost = train_model(point_tuple, epochs, lr, optimizer=optim, test_func=func)
            return history, cost, list_gradient
        
        if optim == "adam+inertial":
            # alpha, beta, initial_guess, lr, epochs, temperature = 0.99, 0.99, point_tuple, 0.001, 800, 0.01 # plot 2
            alpha, beta, initial_guess, lr, epochs, temperature = 0.9, 0.9, point_tuple, 0.01, 800, 0.01
            history, list_gradient, cost = train_model_adam_adamom(alpha, beta, initial_guess, lr, nb_epochs=epochs, func="beale", temperature=temperature)
            return history, cost, list_gradient
        
        if optim == "adamW+inertial":
            # alpha, beta, initial_guess, lr, epochs, temperature = 0.99, 0.99, point_tuple, 0.001, 800, 0.01 # plot 2
            alpha, beta, initial_guess, lr, epochs, temperature, weight_decay = 0.9, 0.9, point_tuple, 0.01, 800, 0.01, 0.001
            history, list_gradient, cost = train_model_adam_adamom(alpha, beta, initial_guess, lr, nb_epochs=epochs, func="beale", temperature=temperature,weight_decay=weight_decay, mode="adamW")
            return history, cost, list_gradient

        if optim == "gd+inertial":
            # offset, epochs, warm_start, eta, alpha, beta = 0.002/0.05, 5000, True, 0.85, 0.99, 0.9
            # offset, epochs, warm_start, eta, alpha, beta = 0.0005, 500, True, 0.85, 0.95, 0.99

            # offset, epochs, warm_start, eta, alpha, beta = 0.001, 5000, True, 0.85, 0.9, 0.9
            offset, epochs, warm_start, eta, alpha, beta = 0.0005, 5000, True, 0.85, 0.995, 0.9
            # offset, epochs, warm_start, eta, alpha, beta = 0.0005, 5200, True, 0.85, 0.999, 0.99
            history, _, cost, gradient_list, _, eigenvalues_list, eigenvalues_estimate_list, eigenvalues_jax_list = inertial_prox_naive([xcoord, ycoord], alpha, beta, epochs, offset, test_func=func, warm_start=warm_start, eta=eta)
            return history, cost, gradient_list, eigenvalues_list, eigenvalues_estimate_list, eigenvalues_jax_list
    
    for i, optim in enumerate(optimizers):
        plt.figure()
        print("*"*20, f"Running optimizer: {optim} for {point_tuple}","*"*20)
        history, cost, gradient_list, eigenvalues_list, eigenvalues_estimate_list, eigenvalues_jax_list = optim_output(optim)
        res_dict = {"history":history, "cost":cost, "gradient_list":gradient_list, "eigenvalues":eigenvalues_list, "eigenvalue_estimates":eigenvalues_estimate_list, "eigenvalue_jax": eigenvalues_jax_list}
        res.append(res_dict)
    return res


def plot_contour_(min_val, max_val, ax, func=None):
    X_grid, Y_grid, Z_grid = xy_meshgrid(min_val, max_val)
    for i in range(len(X_grid)):
        for j in range(len(Y_grid)):
            beta_local = np.array([X_grid[i][j], Y_grid[i][j]])
            function_value = objective_function_plot_xy_independent(beta_local, test_func=func)       
            Z_grid[i, j] = function_value

    # plt.subplots(1,3,1)
    # beale & rosenbrock
    # cp = ax.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap="RdYlBu_r") # beale cmap="RdYlBu_r", "terrain"
    
    # goldstein  & ackley
    cp = ax.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(0, 1, 50), norm=LogNorm(), cmap="RdYlGn") # beale cmap="RdYlBu_r", "terrain"
    for line in cp.collections:
        line.set_linewidth(1.5) #1.5#2.5
    # ax.clabel(cp, inline=2, fontsize=10)
    ax.axis('off')
    ax.set_ylabel('x2')
    ax.set_xlabel('x1')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

def plot_trajectory(min_val, max_val, point_tuple, history, gradient_list, cost, colors, ax, func=None, marker_style_map=None, optimizer="sgd"):
        """ plot for one point """
        # beale & rosenbrock
        xcoord, ycoord = point_tuple
        plot_contour_(min_val, max_val, ax, func=func)
        print("*"*89)
        print("point taken ....", xcoord, ycoord)

        loop_iter = 0
        print("No. of gradient points:", len(history))
        for (i, (beta_i, grad_i)) in enumerate(zip(history, gradient_list)):
            if i == 0:
                ax.scatter([beta_i[0]], [beta_i[1]], s=40, c='black', marker="o", linewidth=5.0, label=str(loop_iter))

            if i%1 == 0:
                print("Epoch:", i, "beta_0: ",[beta_i[0]], "beta_1: " , [beta_i[1]], "beta norm: ", np.linalg.norm(beta_i), "grad norm: ", np.linalg.norm(grad_i), "cost: ", cost[i])
                ax.scatter([beta_i[0]], [beta_i[1]], s=20, c=colors, marker=marker_style_map, linewidth=1, label=str(loop_iter))

            if i == len(history)-1: # last point with black color
                print(i)
                ax.scatter([beta_i[0]], [beta_i[1]], s=70, c=colors, marker="d", linewidth=2.0, label=str(loop_iter))
                loop_iter+=1
        
        # for inertial_sgd, plot 1
        if func == "beale":
            ax.plot(3,0.5, c='black', marker="*", markersize=18)
        if func == "ackley":
            ax.plot(0,0, 'k*', markersize=18)
            ax.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c=colors, linewidth=1)        


def multiple_init(init_points, test_func="beale", optim=None):
    optimizers = ["gd+inertial"]
    map_list = {"beale":[-4.5,4], "ackley":[-4,4]} #[-3,3.5]
    min_val, max_val = map_list[test_func] 

    for i, point_tuple in enumerate(init_points):
        filename = f"{test_func}_{i}_point_figures.png"
        res = compare_against_others(point_tuple, optimizers, func=test_func)
        print("\n", "#"*30, "Plotting", "#"*30)

        if len(res) == 1: # multiple points if multiple optim.
            history, cost, gradient_list, eigenvalues_list, eigenvalues_estimate_list, eigenvalues_jax_list = res[0]['history'], res[0]['cost'], res[0]['gradient_list'], res[0]['eigenvalues'], res[0]['eigenvalue_estimates'], res[0]['eigenvalue_jax']
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(40, 10)) #figsize=(20, 5)
            fig.suptitle(f"Init point: {point_tuple}")

            plot_trajectory(min_val, max_val, point_tuple, history, gradient_list, cost, 'red', ax1, func=test_func, marker_style_map="x", optimizer=optim)        
    
            # Getting only the axes specified by ax[0,0], save single subplot
            extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig('only_coutour.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=300)

            ax2.plot(list(range(len(eigenvalues_list))), eigenvalues_list, label="eigenvalues exact", color="green")
            ax2.plot(list(range(len(eigenvalues_estimate_list))), eigenvalues_estimate_list, label="eigenvalues estimate", color="orange")
            # ax2.plot(list(range(len(eigenvalues_jax_list))), eigenvalues_jax_list, label="eigenvalues jax", color="blue")
            ax2.set(xlabel="iterations", ylabel="Largest Eigenvalue")
            ax2.legend(loc="upper right")
            ax2.set_ylim([-45, 200])
            # ax2.autoscale()

            ax3.plot(list(range(len(cost))), cost, label="cost")
            ax3.plot(list(range(len(gradient_list))), [np.linalg.norm(grad) for grad in gradient_list], label="grad norm", linestyle="--")
            ax3.set(xlabel="iterations", ylabel="Training loss")
            ax3.legend(loc="upper right")
            ax3.set_ylim([-3, max(cost)])
            ax3.axhline(y=min(cost), linestyle="--", color="grey")
            # fig.tight_layout()
            plt.savefig(filename)


# init_points = [(-3, 1.2), (-2,1.1), (0.000001,1)]
# init_points = [(-0.3,2.2),(0, -3.5)]
# init_points = [(-4.3, 1.3)]#[(0, -3.5)]

# beale
# easy_init_points = [(0, -3), (0.00001,1)] # tough_init_points = [(-2, 1.1),(-3, 1.3)]
# init_points = easy_init_points
# multiple_init(init_points, test_func="beale", optim="gd+inertial")

# ackley
easy_init_points = [(-1.5,3.5)]
init_points = easy_init_points
multiple_init(init_points, test_func="ackley", optim="gd+inertial")
        