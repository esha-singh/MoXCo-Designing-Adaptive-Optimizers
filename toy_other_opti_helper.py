import os
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from collections import defaultdict

import torch
from torch import optim 
from loss_function_graphs import beale_function, rosenbrock_function, goldstein_price_function, ackley_function
from toy_helper_2d import plot_contour

path = os.path.abspath("./../../../")
sys.path.append(path)
# from HessianFlow import hessianflow as hf
# from aligator_resnet import init_experts, run_aligator, get_awake_set
device = "cuda" if torch.cuda.is_available() else "cpu"
    

def adaMoM_pre_game(loss_grad_snapshot, near_expected_grad, z_i_norm_square_list, variance_for_sigma_inside_aligator, Y_mean_estimate, y_mean_square_estimate):
    # near_expected_grad is outside element
    z_i = np.subtract(loss_grad_snapshot, near_expected_grad) # element wise subracting to get noise estimate

    k = np.linalg.norm(z_i)**2
    z_i_norm_square_list.append(k)
    z_i_norm_square = np.mean(z_i_norm_square_list)

    Y_sigma_inside_aligator = 2*np.dot(z_i, loss_grad_snapshot) + k
    print("Yi: ", 2*np.dot(z_i, loss_grad_snapshot))
    
    # sigma the one used in aligator
    Y_mean_estimate = 0.01*np.linalg.norm(Y_sigma_inside_aligator) + 0.99*Y_mean_estimate
    y_mean_square_estimate = 0.01*np.linalg.norm(Y_sigma_inside_aligator)**2 + 0.99*y_mean_square_estimate
    variance_for_sigma_inside_aligator = y_mean_square_estimate - Y_mean_estimate**2
    return near_expected_grad, z_i_norm_square, z_i_norm_square_list, variance_for_sigma_inside_aligator, Y_mean_estimate, y_mean_square_estimate
                
            
def objective_function_edited(beta_hat, FLAG=None):
    """ Initial experiments: fixed data based loss for simpler plotting. This removes dependence on data"""
    loss = 0
    if FLAG == 'beale':
        x = beta_hat 
        loss = (1.5 - x[0] + x[0] * x[1]).pow(2) + (2.25 - x[0] + x[0] * x[1] * x[1]).pow(2) + 2.625 - x[0] + (x[0] * x[1] * x[1] * x[1]).pow(2)
        
    if FLAG == "rosenbrock":
        x = beta_hat
        a, b = 1, 100
        loss = (a - x[0]).pow(2) + b*((x[1] - x[0].pow(2)).pow(2))

    if FLAG == "goldstein-price":
        x = beta_hat
        loss = (1 + ((x[0] + x[1] + 1).pow(2)) * (19 - 14*x[0] + 3*(x[0]*x[0]) - 14*x[1] + 6*x[0]*x[1] + 3*x[1]*x[1])) * (30 + (2*x[0] - 3*x[1]).pow(2)*(18 - 32*x[0] + 12*x[0]*x[0] + 48*x[1] - 36*x[0]*x[1] + 27*x[1]*x[1]) -8.693)
        loss = (1/2.427)*torch.log(loss)

    if FLAG == "ackley":
        x = beta_hat
        loss = -20 * torch.exp(-0.2 * torch.sqrt(0.5*(x[0]*x[0] + x[1]*x[1]))) - torch.exp(0.5*(torch.cos(2*math.pi*x[0]) + torch.cos(2*math.pi*x[1]))) + math.e + 20
    return loss
    
def train_model(initial_guess, nb_epochs, lr, optimizer="sgd", test_func="beale"):
    ''' training code with optimizer from pytorch lib
    check which gradient list used.
    '''
    model_named_parameters = {}
    model_named_parameters["l0"]=  torch.tensor(initial_guess[0],  dtype=torch.float32, requires_grad=True, device=device)
    model_named_parameters["l1"] = torch.tensor(initial_guess[1],  dtype=torch.float32, requires_grad=True, device=device)
    prev = {}
    for n, p in model_named_parameters.items():
        prev[n] = p.clone()

    if optimizer == "sgd":
        optimizer = optim.SGD([model_named_parameters["l0"], model_named_parameters["l1"]], lr=lr, momentum=0)
    if optimizer == 'adam':
        optimizer = optim.Adam([model_named_parameters["l0"], model_named_parameters["l1"]], lr=lr, betas=[0.9,0.999])
    if optimizer == 'adamW':
        import AdamW
        optimizer = AdamW([model_named_parameters["l0"], model_named_parameters["l1"]], betas=[0.9,0.999], lr=lr, weight_decay=0.01)

    x_prev = 0
    beta_history, list_gradient, list_corrected_gradient, losses, losses_list, prob = [], [], [], [], [], []
    loss= 0
    for j in range(0, nb_epochs):   
        # forward pass
        beta_hat = [x for _,x in model_named_parameters.items()]
        loss = objective_function_edited(beta_hat, FLAG=test_func)

        optimizer.zero_grad()
        
        with torch.no_grad():
            b_current = np.array([k.item() for _,k in model_named_parameters.items()])
            b_current_ordered = np.array((b_current[0], b_current[1])) #reorder (bias, weight) #1, 0
        beta_history.append(b_current_ordered)
        losses.append(loss.item())
        loss.backward()
        
        # Store the gradient
        with torch.no_grad():
            grad = np.zeros(2)
            for index_p, (_,p) in enumerate(model_named_parameters.items()):
                grad[index_p] = p.grad.detach().data 
            grad_ordered = np.array((grad[1], grad[0]))#reorder (bias, weight)
            list_gradient.append(grad_ordered)

        # losses.append(loss.item())
        # STEP UPDATE
        optimizer.step()

        # losses_list.append(np.mean(losses))
        # print("loss avg: ", np.mean(losses))
        losses_list.append(loss.item())
        print("loss avg: ", loss.item())
        if abs(x_prev - loss.item()) <= 1e-5 or np.linalg.norm(grad_ordered) <= 1e-5:#5e-4:#1e-5:#5e-5: #5e-4
        # if abs(x_prev - loss.item()) <= 1e-5 or np.linalg.norm(grad_ord) <= 1e-3:#5e-4:#1e-5:#5e-5: #5e-4
            print("BREAKING>>>")
            break
        x_prev = loss.item()
    return beta_history, list_gradient, losses_list

def train_model_adaMoM(initial_guess, nb_epochs, lr, optimizer="adam", test_func="beale", resetting_window=0.868, temperature=1):
    ''' training code with optimizer from pytorch lib
    check which gradient list used.
    '''
    WARM_START = False
    near_expected_grad_list, betas = [], [0.99, 0.999]

    model_named_parameters = {}
    model_named_parameters["l0"]=  torch.tensor(initial_guess[0],  dtype=torch.float32, requires_grad=True, device=device)
    model_named_parameters["l1"] = torch.tensor(initial_guess[1],  dtype=torch.float32, requires_grad=True, device=device)
    prev = {}
    for n, p in model_named_parameters.items():
        prev[n] = p.clone()

    mom_reset = 0.9
    if optimizer == "sgd":
        optimizer = optim.SGD([model_named_parameters["l0"], model_named_parameters["l1"]], lr=lr, momentum=mom_reset)
    if optimizer == 'adam':
        optimizer = optim.Adam([model_named_parameters["l0"], model_named_parameters["l1"]], betas=betas, lr=lr)
    if optimizer == 'adamW':
        import AdamW
        optimizer = AdamW([model_named_parameters["l0"], model_named_parameters["l1"]], betas=betas, lr=lr)

    
    beta_history, list_gradient, losses, losses_list, prob = [], [], [], [], []
    loss, grad_list = 0, []
    n, pool = nb_epochs, []
    # pool_size, pool = init_experts(pool, n)
    # print("pool len:", len(pool), "Pool size:", pool_size)
    Y_mean_estimate, y_mean_square_estimate, near_expected_grad_list, z_i_norm_square_list = 0,0, np.zeros(len(model_named_parameters), dtype='float64'), [0]
    max_grad, z_i_norm_square, variance_for_sigma_inside_aligator = 0, 0, 0.1
    x_prev = 0
    for j in range(0, nb_epochs):   
        # forward pass
        beta_hat = [x for _,x in model_named_parameters.items()]
        loss = objective_function_edited(beta_hat, FLAG=test_func)

        optimizer.zero_grad()
        
        with torch.no_grad():
            b_current = np.array([k.item() for _,k in model_named_parameters.items()])
            b_current_ordered = np.array((b_current[0], b_current[1])) #reorder (bias, weight) #1, 0
        beta_history.append(b_current_ordered)
        
        losses.append(loss.item())
        loss.backward()
        
        # Store the gradient
        with torch.no_grad():
            grad = np.zeros(len(model_named_parameters))
            for index_p, (_,p) in enumerate(model_named_parameters.items()):
                grad[index_p] = p.grad.detach().data 
            grad_ordered = np.array((grad[1], grad[0]))#reorder (bias, weight)
            list_gradient.append(grad_ordered)

        # losses.append(loss.item())
        optimizer.step()

        # --------------------------- ADAMOM --------------------------------------------------------------------------- #
        # calculating the aligator noise estimates
        if WARM_START == False:
            # grad_i = np.zeros(len(model_named_parameters))
            # counter, grad_norm = 0, 0
            # grad_ord = None
            # for _,p in model_named_parameters.items():
            #     tmp = p.grad.data.cpu().detach().numpy()
            #     # l2_norm = np.linalg.norm(tmp)
            #     grad_i[counter]= tmp
            #     counter+=1
            # grad_ord = np.array((grad_i[0], grad_i[1]))
            with torch.no_grad():
                grad_list = np.zeros(2)
                grad_ord, grad_norm = None, 0
                c=0
                for _,p in model_named_parameters.items():
                    grad_list[c] = p.grad.data.cpu().detach().numpy()
                    c+=1
                grad_ord = np.array((grad_list[0], grad_list[1]))

                # grad_list.append(l2_norm)
                # grad_norm += l2_norm**2
                # count+=1 
            # grad_i_norm_square = np.linalg.norm(grad_i)**2
            # max_grad = max(max_grad, grad_norm)

            # aligator_input = grad_i_norm_square #- z_i_norm_square
            # time_range, B, delta = nb_epochs, max_grad, pow(10,-2)
            # denoised_grad_norm_squared = run_aligator(time_range, j, aligator_input, pool, pool_size, variance_for_sigma_inside_aligator, B, delta)
            # print("-"*40, denoised_grad_norm_squared, "grad_norm: ", grad_i_norm_square, "Bias z_i inside:", z_i_norm_square, "Aligator i/p: ", aligator_input, variance_for_sigma_inside_aligator, max_grad)

            # near_expected_grad = near_expected_grad_list
            # near_expected_grad, z_i_norm_square, z_i_norm_square_list, variance_for_sigma_inside_aligator, Y_mean_estimate, y_mean_square_estimate = adaMoM_pre_game(grad_i, near_expected_grad, z_i_norm_square_list, variance_for_sigma_inside_aligator, Y_mean_estimate, y_mean_square_estimate)

            # edge_of_stability_bound = 2/(lr)
            # grad_norm = denoised_grad_norm_squared/nb_epochs
            grad_norm = np.linalg.norm(grad_ord)**2
            main_term = 0.01*float(loss.item()) +  grad_norm*0.01
            
            # vec = hessian_eigenvalue_estimation([model_named_parameters["l0"], model_named_parameters["l1"]], objective_function_edited, beta_hat)
            f_x =  main_term #+ vec/(edge_of_stability_bound)
            prob = np.exp(-main_term)
            # print("--", prob, "->->->", float(loss.item()), "Aligator grad: ", grad_norm, "Epoch: ", j, "fx: ", f_x)
            print("--", prob, "->->->", float(loss.item()), "Epoch: ", j, "fx: ", main_term, "grad_norm: ", grad_norm)
            # near_expected_grad_list = np.mean(grad_list)

            if prob > resetting_window:
                WARM_START=True
                print("**********************WARM START**********************")
                betas = [0.96, 0.3]
                # mom_reset = 0.2
                optimizer.param_groups[0]['betas'] = [betas[0], betas[1]]
                # optimizer.param_groups[0]['momentum'] = mom_reset
        # --------------------------- ADAMOM --------------------------------------------------------------------------- #
        losses_list.append(np.mean(losses))
        print("loss avg: ", np.mean(losses), optimizer.param_groups[0]['betas'], abs(x_prev - loss.item()))
        # print("loss avg: ", np.mean(losses), optimizer.param_groups[0]['momentum'])
        if abs(x_prev - loss.item()) <= 1e-5 or np.linalg.norm(grad_ord) <= 1e-5:#5e-4:#1e-5:#5e-5: #5e-4
        # if abs(x_prev - loss.item()) <= 1e-5 or np.linalg.norm(grad_ord) <= 1e-3:#5e-4:#1e-5:#5e-5: #5e-4
            print("BREAKING>>> ")
            break
        x_prev = loss.item()

    return beta_history, list_gradient, losses_list

def run(theta_i, epochs, lr, func="beale", adaptive_momentum = False, resetting_window=None, temperature=None):
    """ theta_i: slope, intercept: b0, b1. It is linked to NN module. Slope is param1 & intercept param2
        1. inside train_model everything is reordered to bias, weight: intecept, slope.
    """
    if adaptive_momentum == True:
        history, list_gradient, cost = train_model_adaMoM(theta_i, epochs, lr, optimizer="adam", test_func="beale", resetting_window=resetting_window, temperature=temperature)
    else:
        history, list_gradient, cost = train_model(theta_i, epochs, lr, optimizer="adam", test_func="beale")
    theta = history[-1] 
    print("Final point: {:.2f}, {:.2f} {:d}".format(theta[0], theta[1], 50))
    return history, theta, cost, list_gradient



# ------------------------------------------------------- all code below this is for testing only --------------------------------------------------------#
def plot_gradient_flow(min_val, max_val, epochs, lr, file_name="2d_beale_gradient.png", func=None, adaptive_momentum=False, resetting_window=None, temperature=None):
    """ plot for one point first"""
    X_grid, Y_grid, filename, ax = plot_contour(min_val, max_val, file_name, func)

    if func == "rosenbrock":
        passive_list = [(-3, 2), (-1.8, 2), (1.5, 0.5)]
        list_of_points = [(3, 4.8)]
    elif func == "beale":
        # 0, 1 saddle point for beale
        # list_of_points = [(0.3, 0.1)]#[(-0, 3.1)]#[(0.4,-1), (0.23, 0.9),(0.5, -0.5),(2,0)]#(0.5, -0.5),(0.23, 0.9), [(-3, 1.2)]#[(0, -4)]#[(-2, 1.3)]#[(0, -3)]#[(0.5, 2)]#[(0.5, 2)]#[(-1,-2), (2,2), (0.4, -1), (0.23, 0.9), (-3, 3)] # add -2, 1
        passive_list = [(-4.3, 1.3),(-0.5,1.5), (-4.3, 1.3), (-3, 1.2)]
        list_of_points = [(-2, 1.3)]#[(0,1)]
    elif func == "goldstein-price":
        passive_list = [(0,0), (-1,3.5), (0,3.5), (0.1,-0.1),(1, 0.7), (1, 1.5), (-2, 2)]
        list_of_points = [(-1,3.5)]
    elif func == "ackley":
        passive_list = [(3,0), (-2,3), (-1,-2), (-1.5,-3.5), (-1,3.3)]
        list_of_points = [(-2,-2)]
    color = ["red", "pink", "orange"]
    for xcoord, ycoord in list_of_points:
        print("*"*89)
        print("point taken ....", xcoord, ycoord)
        history, _, cost, gradient_list = run([xcoord, ycoord], epochs, lr, func=func, adaptive_momentum=adaptive_momentum, resetting_window=resetting_window, temperature=temperature)#inertial_prox_naive(x, y, [xcoord, ycoord], lr, br, batch_size)
        loop_iter = 0
        print("No. of gradient points:", len(history))
        for (i, (beta_i, grad_i)) in enumerate(zip(history, gradient_list)):
            if i%1 == 0:
                print("Epoch:", i, "beta_0: ",[beta_i[0]], "beta_1: " , [beta_i[1]], "beta norm: ", np.linalg.norm(beta_i), "grad norm: ", np.linalg.norm(grad_i), "cost: ", cost[i])
                plt.scatter([beta_i[0]], [beta_i[1]], s=100, c='red', marker="x", linewidth=2.0, label=str(loop_iter)) #mediumpurple
                if i == 0:
                    plt.arrow(beta_i[0], beta_i[1], - 10 * 0.001 * grad_i[1], - 10 * 0.001 * grad_i[0], color='green')
                else:
                    plt.arrow(beta_i[0], beta_i[1], - 10 * 0.001 * grad_i[1], - 10 * 0.001 * grad_i[0], color='darkblue')

            if i == len(history)-1:
                print(i)
                plt.scatter([beta_i[0]], [beta_i[1]], s=30, c='black', linewidth=5.0, label=str(loop_iter))
                loop_iter+=1

    if func == "rosenbrock":
        plt.plot(1,1, 'r*', markersize=18)
        # plt.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c="black", linewidth=1)
    elif func== "beale":
        plt.plot(3,0.5, 'b*', markersize=18)
    elif func == "goldstein-price":
        plt.plot(0,-1, 'b*', markersize=18)
    elif func == "ackley":
        plt.plot(0,0, 'b*', markersize=18)
        # plt.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c="black", linewidth=1)
    plt.savefig(filename)
    plt.close()

# min_val, max_val = -4, 4
# epochs, lr, resetting_window, temperature = 2000, 0.01, 0.68, 1 #0.82
# plot_gradient_flow(min_val, max_val, epochs, lr, file_name="2d_beale_gradient_adam.png", func="beale", adaptive_momentum=True, resetting_window=resetting_window, temperature=temperature)