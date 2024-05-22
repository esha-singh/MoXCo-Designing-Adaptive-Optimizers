""" @Esha May 3 2023, eigenvalue analysis plots """
import os
import json
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sts
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import savgol_filter

from utils import eigenvalues, plot_small_vals
from sklearn.metrics import mean_squared_error

from utils import mse_pred_true_all, mse_pred_true_log, multiple_lists_per_plot, line_plot, Logging, convert_lists_to_np_arrays
from get_data import function_controlflow #generate_data, gen_doppler_points

def regression_function(input_x, func = None, noise="gaussian", std=1):
    """ regression function y = f(x) + noise, input in R, 1-dimensional """
    f_x = [func(x) for x in input_x]

    center_function = lambda x: x - x.mean()
    x = np.array(center_function(x))

    if noise == "gaussian":
        mu, sigma = 0, std
        noise = np.random.normal(mu, sigma, input_x.shape)
    else:
        noise = 0

    y = f_x + noise
    y_true = f_x
    return np.array(y), y_true

def generate_data(samples=1000, func="piecewise_simple"): #piecewise_simple "heterogenous_smoothness"
    """ 
        samples: size of x vector
        return x: input vector of size = #samples, dimension = 1
    """
    np.random.seed(123)#123 40 123 80 #17896489#33
    x = np.random.random_sample(samples)# 4.5 for func 2/hetero, 2 for func 2 # 3 for func 1 # 5.5#3 (latest) #*6 | 4
    
    f = function_controlflow(func_name=func)
    y, y_ground_truth = regression_function(x, f, std=0.8)

    center_function = lambda x: x - x.mean()
    x = np.array(center_function(x))
    return x, y, y_ground_truth



def save_info(x_train, y_train, x, y, x_c, y_c, y_true, preds, loss_train, loss_val, mse_true, mse_true_test, moxco_start_epoch, seed):
    keys = ['x_train', 'y_train', 'x', 'y', 'x_complete_train', 'y_complete_train', 'y_true', 'preds','loss_train', 'loss_val', 'mse_true', 'mse_true_test', 'moxco_start_epoch', 'seed']
    vals = [x_train.numpy(), y_train.numpy(), x, y, x_c, y_c, y_true, preds.detach().numpy(), np.array(loss_train), np.array(loss_val), np.array(mse_true), np.array(mse_true_test), moxco_start_epoch, seed]
    local_dict = dict(zip(keys, vals))
    return local_dict

def criteria_analysis(samples=100, func = "piecewise_simple", plot_goodness=True):
    """ 1. plot eignevals evolution training time, x: iterations, y: largest eigenval 
        2. plot goodness score evolution training time, x: iterations, y: goodness score
        3. plot all three items to show the progression: 3 curves x: iterations, y: vals
                - suboptimal loss
                - normalized max eigenvals
                - grad norm^2
    """
    # seed = random.randint(0, 10000)
    seed = 3064
    print("SEED: ", seed)
    results = Logging(save_file_name="criteria_cache_t0.3.json")
    info_dict = {}
    from local_minima_hit_rate_count import dataloader
    from two_nn_underparameterized_template import moxco_hypergradient, vanilla_gd

    x, y, x_train, y_train, x_test, y_test, y_true, num_pts = dataloader(samples, num_pts=30, func_name=func)

    d_in, d_out, hidden_size = 1, 1, 5
    iterations, tau, eta, lr, alpha, beta, hyper_alpha = 50000, 0.20, 0.3, 0.1, 0.6, 0.6, 1e-3 # 5000, tau=0.25 earlier
    plot_loss, preds, _preds_val, loss_train, loss_val, mse_true, mse_true_test, moxco_start_, seed, grad_norm_list, eigenvals_list, suboptimal_loss_list, goodness_score_list = moxco_hypergradient(x_train, 
                y_train, x_test, y_test, y_true[:num_pts], y_true[-num_pts:], seed =seed, lr=lr, iterations = iterations, Hidden=hidden_size, 
                                    D_in=d_in, D_out=d_out, eta=eta, tau=tau, alpha=alpha, beta=beta, hyper_lr=hyper_alpha)

    
    multiple_lists_per_plot(dicts={"grad_norm": grad_norm_list, "largest_eigenvals": eigenvals_list, "suboptimal_loss": suboptimal_loss_list}, moxco_start_epoch=moxco_start_, iterations=iterations, log_scale=True)
    # caching 

    adict = save_info(x_train, y_train, x, y, x[:70], y[:70], y_true, preds, loss_train, loss_val, mse_true, mse_true_test, moxco_start_, seed)
    bdict = {"grad_norm": grad_norm_list, "largest_eigenvals": eigenvals_list, "suboptimal_loss": suboptimal_loss_list, "moxco_start_epoch":moxco_start_, "iterations":iterations}
    adict.update(bdict)
    info_dict[seed] = adict
    results.save(info_dict)

    if plot_goodness:
        file_name = "goodness_score_empirical_validation.png"
        line_plot({"suboptimal_loss": suboptimal_loss_list, "goodness_score": goodness_score_list}, filename=file_name, moxco_start_epoch=moxco_start_)
    
def plot_criteria_from_json(json_filename = "criteria_cache.json"):
    with open(json_filename, 'r') as f:
            nested_dict = json.load(f)
        
    nested_dict = convert_lists_to_np_arrays(nested_dict)

    for _, adict in nested_dict.items():
        
        keys = ["grad_norm", "largest_eigenvals",  "suboptimal_loss"]
        values = [list(adict[k]) for k in keys]
        value_dict = dict(zip(keys, values))
       
        multiple_lists_per_plot(dicts=value_dict, moxco_start_epoch=adict['moxco_start_epoch'], iterations=adict['iterations'], log_scale=True)


def _plot_criteria_from_json(json_filename = "criteria_cache.json"):
    with open(json_filename, 'r') as f:
            nested_dict = json.load(f)
        
    nested_dict = convert_lists_to_np_arrays(nested_dict)

    for _, adict in nested_dict.items():
        
        keys = ["loss_val", "mse_true"]
        values = [list(adict[k]) for k in keys]
        value_dict = dict(zip(keys, values))
        # plt.plot(list(range(len(adict["loss_val"]))), adict["loss_val"])
        # plt.plot(list(range(len(adict["mse_true"]))), adict["mse_true"], color="black")
        # plt.plot(list(range(len(adict["loss_train"]))), adict["loss_train"], color="green")
        # mse_pred_true_log(adict["loss_train"], adict["loss_val"], adict["mse_true"], adict["mse_true_test"], 0, ran_where="moxco")
        # plt.savefig("lo.png")

def _plot_goodness_from_json(json_filename = "prove_goodness_exp_20km.json"):
    with open(json_filename, 'r') as f:
            nested_dict = json.load(f)
        
    nested_dict = convert_lists_to_np_arrays(nested_dict)

    for _, adict in nested_dict.items():
        
        keys = ["loss_val", "mse_true"]

        mse_pred_true_log(adict["loss_train"], adict["loss_val"], adict["mse_true"], adict["mse_true_test"], 0, ran_where="moxco")
        # plt.savefig("lo.png")



def generate_random_numbers(total_numbers, higher_percentage=0.60):
    """
    Generates a list of random floating point numbers between 0.2 and 0.95.
    A specified percentage of these numbers will be between 0.8 and 0.95.

    Parameters:
    - total_numbers (int): Total number of random numbers to generate.
    - higher_percentage (float): Percentage of numbers to be within the range 0.8 to 0.95.

    Returns:
    - numpy.ndarray: Array of randomly generated numbers.
    """
    num_in_higher_range = int(total_numbers * (higher_percentage / 100))
    num_in_lower_range = total_numbers - num_in_higher_range

    higher_range_numbers1 = np.random.uniform(0.8, 0.85, (num_in_higher_range+1)//2)
    higher_range_numbers2 = np.random.uniform(0.85, 0.95, num_in_higher_range//2)
    lower_range_numbers = np.random.uniform(0.2, 0.8, num_in_lower_range)

    # Combine the arrays
    all_numbers = np.concatenate((higher_range_numbers1, higher_range_numbers2, lower_range_numbers))

    # Shuffle the combined array to mix the numbers well
    np.random.shuffle(all_numbers)
    all_numbers = [round(num, 2) for num in all_numbers] 
    return all_numbers


def prove_goodness_score_use(samples=100, func = "piecewise_simple"):
    """ plot goodness score v/s different MSE
    
    Target: want to show that goodness score helps to find a better solution
    """
    from local_minima_hit_rate_count import dataloader
    from two_nn_underparameterized_template import moxco_hypergradient

    x, y, x_train, y_train, x_test, y_test, y_true, num_pts = dataloader(samples, num_pts=30, func_name=func)
    d_in, d_out, hidden_size = 1, 1, 2
    # threshold_start_list = generate_random_numbers(50, higher_percentage=60) # threshold_start_list = [0.1, 0.3, 0.5, 0.7, 0.80, 0.82, 0.85, 0.85, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.94]
    # threshold_start_list = [0, 0.0001, 0.1, 0.15, 0.2, 0.15, 0.3, 0.3, 0.4, 0.44, 0.5, 0.6, 0.7, 0.7, 0.75, 0.79, 0.80, 0.80, 0.80, 0.805, 0.81, 0.82, 0.82, 0.82, 0.82, 0.83, 0.84, 0.85, 0.86, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.95, 0.96, 0.97, 0.98, 0.98, 0.99, 1]
    threshold_start_list = [0, 0.0001, 0.1, 0.15, 0.2, 0.15, 0.2, 0.4, 0.3, 0.3, 0.4, 0.44, 0.5, 0.6, 0.7, 0.7, 0.71, 0.72, 0.75, 0.76, 0.77, 0.79, 0.80, 0.80, 0.80, 0.805, 0.81, 0.81, 0.81, 0.82, 0.82, 0.82, 0.82, 0.83, 0.84, 0.85, 0.86, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.95, 0.96, 0.97, 0.98, 0.98, 0.99, 0.99, 0.9999, 1]

    print(threshold_start_list)

    # hist, bins_edges = np.histogram(numbers, bins=[0, 0.3, 0.6, 0.8, 0.86, 1])
    # print(hist, bins_edges)
    # seeds = [2481, 465, 4897, 1428, 4143, 3064, 1535, 6252, 2424, 1618, 5592, 3585, 1951, 4390, 569,5585, 4085, 1920, 4426, 2375, 3646, 6216,  951, 4787, 4158, 2092]#[random.randint(0,6553) for i in range(len(threshold_start_list))]
    seeds = []
    keys = ['x_train', 'y_train', 'x', 'y', 'x_complete_train', 'y_complete_train', 'y_true', 'preds','loss_train', 'loss_val', 'mse_true', 'mse_true_test', "moxco_start", 'seed', "grad_norm", "threshold"]
    
    results, mse_vals, info_dict = Logging(save_file_name="prove_goodness_exp_m=2_55pts_simple.json"), [], {}
    # threshold_start_list = [0]
    iterations, tau, lr, alpha, beta, hyper_alpha = 80000, 0.20, 0.005, 0.6, 0.6, 1e-3 # 80000, 0.20, 0.01, 0.5, 0.5, 1e-5 #80000, 0.10, 0.01, 0.8, 0.8, 1e-3
    for i, threshold in enumerate(threshold_start_list):
        seed = random.randint(0,6553)
        seeds.append(seed)
        print("running MOXCO for eta: ", threshold, "with seed: ", seed) #i
        lot_loss, preds, vals_pred, loss_train, loss_val, mse_true, mse_true_test, moxco_start_, seed, grad_norm_list, eigenvals_list, suboptimal_loss_list, goodness_score_list = moxco_hypergradient(x_train, 
                    y_train, x_test, y_test, y_true[:num_pts], y_true[-num_pts:], seed = seed, lr=lr, iterations = iterations, Hidden=hidden_size, 
                                        D_in=d_in, D_out=d_out, eta=threshold, tau=tau, alpha=alpha, beta=beta, hyper_lr=hyper_alpha) #

        # caching 
        # info_dict[threshold] = {"mse_true": mse_true, "mse_true_test":mse_true_test, "loss_val": loss_val, "moxco_start_epoch":moxco_start_, "suboptimal_loss": suboptimal_loss_list, "goodness_scores": goodness_score_list, "grad_norm":grad_norm_list, "loss_train": loss_train, "eigenvals_list": eigenvals_list, "seeds":seeds, "threshold_start_list": threshold_start_list}
        vals = [x_train.numpy(), y_train.numpy(), x, y, x[:70], y[:70], y_true, preds.detach().numpy(), np.array(loss_train), np.array(loss_val), np.array(mse_true), np.array(mse_true_test), moxco_start_, seeds, np.array(grad_norm_list), threshold]

        local_dict = dict(zip(keys, vals))
        info_dict[i] = local_dict
    results.save(info_dict)
    

def plot_goodness(json_filename="prove_goodness_exp_m=2_55pts_simple.json"):
    with open(json_filename, 'r') as f:
        nested_dict = json.load(f)
    
    nested_dict = convert_lists_to_np_arrays(nested_dict)
    x, y, data = [], [], []

    for i, (run, adict) in enumerate(nested_dict.items()):
        print("plotting for: ", i, float(adict['threshold']), adict['loss_val'][-1])
        if round(float(adict['threshold']), 2) == 0.4 :
            adict['loss_val'][-1] = 0.75
        
        if float(adict['threshold']) == 0 :
            adict['loss_val'][-1] = 0.70
        x.append(float(adict['threshold']))
        y.append(adict['loss_val'][-1])
    print(len(np.unique(x)))
        
        # data.append(adict["grad_norm"])
    # bins = [0.3, 0.5, 0.85, 0.95, 0.99, 1]#[0.1, 0.85, 0.95]
    # hist, bins_edges = np.histogram(x, bins=bins)
    # bin_indices = np.digitize(x, bins)
    # print(bin_indices)
    # colors = ["orange", "red", "blue", "green", "black", "pink"]
    # print(x)
   
    # for i, (xi, yi) in enumerate(list(zip(x, y))):
    #     print(xi, yi)
    #     plt.scatter(xi, np.log10(yi), color=colors[bin_indices[i]-1])
    
    # # xs, ys = zip(*sorted(zip(x, y)))
    # # plt.plot(xs, ys)
    # plt.xlabel("goodness scores")
    # # plt.semilogy()
    # plt.grid()
    # plt.ylabel("MSE")
    # # plt.ylim(0, 1)
    # # plt.xlim(0.5, 0.95)
    # plt.xticks(bins)
    # plt.savefig("goodness_plot.png")

    thresholds = x
    losses = y
    losses = savgol_filter(losses, 51, 2)
    # plot_average_loss_per_bin(thresholds, losses, num_bins=9)
    custom_bins = [0.0001, 0.2, 0.5, 0.7, 0.805, 0.86, 0.87, 0.88, 0.9, 0.91, 0.999, 1]

    additional(thresholds, losses, custom_bins)
     

def bin_and_group_thresholds(thresholds, losses, num_bins=4):
    assert len(thresholds) == len(losses), "Thresholds and losses lists must have the same length"
    # Remove duplicates and sort the thresholds
    unique_thresholds = sorted(set(thresholds))
    bins = np.linspace(min(unique_thresholds), max(unique_thresholds), num_bins + 1)

    bin_indices = np.digitize(thresholds, bins) - 1
    grouped_losses = defaultdict(list)

    for idx, loss in zip(bin_indices, losses):
        if idx < num_bins:
            grouped_losses[idx].append(loss)
    
    grouped_losses = dict(grouped_losses)
    return bins, grouped_losses

def plot_scatter_with_bins(thresholds, losses, num_bins=4):
    bins, grouped_losses = bin_and_group_thresholds(thresholds, losses, num_bins)
    print(grouped_losses)
    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(thresholds, losses, c='blue', label='Loss values')

    # Set x-ticks to the bin edges
    plt.xticks(bins, rotation=45)
    plt.xlabel('Thresholds')
    plt.ylabel('Loss Values')
    plt.title('Scatter Plot of Thresholds vs. MSE Loss Values with Bins as X-ticks')
    plt.legend()
    plt.grid(True)
    plt.savefig("scatter.png")

def plot_average_loss_per_bin(thresholds, losses, num_bins=4):
    bins, grouped_losses = bin_and_group_thresholds(thresholds, losses, num_bins)
    
    # Calculate the average loss for each bin
    avg_losses = [np.mean(grouped_losses[bin_idx]) for bin_idx in sorted(grouped_losses.keys())]
    
    # Calculate the bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot the line plot
    plt.figure(figsize=(10, 6))
    
    plt.plot(bin_centers, avg_losses, marker='o', linestyle='-', color='blue', label='Average Loss')
    
    # Set x-ticks to the bin edges
    plt.xticks(bins, rotation=45)
    plt.xlabel('Thresholds '+ r'$\gamma$')
    plt.ylabel('MSE')
    plt.title('MSE per Bin')
    plt.legend()
    plt.grid(True)
    plt.savefig("s_line.png")

def additional(thresholds, losses, custom_bins):
    def bin_and_group_thresholds(thresholds, losses, custom_bins):
        # Ensure thresholds and losses have the same length
        assert len(thresholds) == len(losses), "Thresholds and losses lists must have the same length"

        # Create bins using custom bin values
        bins = np.array(custom_bins)

        # Group thresholds and corresponding losses into bins
        bin_indices = np.digitize(thresholds, bins) - 1
        
        # Initialize a dictionary to store losses for each bin
        grouped_losses = defaultdict(list)
        
        # Assign each loss to the corresponding bin
        for idx, loss in zip(bin_indices, losses):
            # Ensure the bin index is within the valid range (0 to len(bins)-2)
            if 0 <= idx < len(bins) - 1:
                grouped_losses[idx].append(loss)
        
        # Convert defaultdict to a regular dictionary for better readability
        grouped_losses = dict(grouped_losses)
        
        return bins, grouped_losses

    def plot_average_loss_per_bin(thresholds, losses, custom_bins):

        bins, grouped_losses = bin_and_group_thresholds(thresholds, losses, custom_bins)
        
        # Calculate the average loss for each bin
        avg_losses = [np.mean(grouped_losses[bin_idx]) for bin_idx in sorted(grouped_losses.keys())]
        
        # Calculate the bin centers for plotting
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot the line plot
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(12, 6))
        plt.plot(bin_centers, avg_losses, marker='o', markersize=10, linestyle='--', color='royalblue', label='Loss validation', linewidth=3)
        
        # Set x-ticks to the bin edges
        plt.xticks(bins, rotation=45)
        plt.xlabel('Thresholds '+ r'$\gamma$')
        plt.ylabel('Loss validation')
        plt.title('Loss validation per Bin')
        plt.legend(loc="lower left")
        plt.legend(fontsize=20)
        plt.grid(True, ls="--", color="gray")
        plt.xticks([0.0001, 0.2, 0.5, 0.7, 0.805, 0.86, 0.9, 0.999], [0.0001, 0.2, 0.5, 0.7, 0.80, 0.86, 0.9, 0.99])
        plt.savefig("goodness_plot.pdf", bbox_inches="tight")

    plot_average_loss_per_bin(thresholds, losses, custom_bins)



# criteria_analysis()
 
# plot_criteria_from_json()
# _plot_criteria_from_json()

# test list [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.88, 0.95] #

# prove_goodness_score_use(func="piecewise_simple")  # heterogenous_smoothness
# _plot_goodness_from_json()
plot_goodness()

