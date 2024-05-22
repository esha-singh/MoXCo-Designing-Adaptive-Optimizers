"""
Task: For different inits of parameters determine how many times 
moxco and vanilla version reach a local minima within max possible
epochs.

results: moxco should have the 
# times reached local minima by moxco/total exps > # times reached local minima by vanilla/total exps > 
"""
import os
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as sts
import matplotlib.pyplot as plt
from utils import eigenvalues, plot_small_vals
from sklearn.metrics import mean_squared_error

from utils import mse_pred_true_all, mse_pred_true_log
from get_data import generate_data, gen_doppler_points
from synthetic_exp2 import scoring, timestamp, RES_DIR


def convert_lists_to_np_arrays(data):
    if isinstance(data, dict):
        return {k: convert_lists_to_np_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return np.array(data)
    return data

def visualize_boundaries_latest(x_complete_train, y_complete_train, x_train, y_train, preds, x, 
                                y_ground_truth = None, nn_fit_line = True, 
                                interpolate_data=True, name=None, path=None):
    plt.figure(figsize=(7, 4))
    if nn_fit_line:
        xs, ys = zip(*sorted(zip(x_train, preds)))
        plt.plot(xs, ys, color="black", label="fitted")
    else:
        plt.scatter(x_train, preds, color="black")

    if interpolate_data:
        plt.scatter(x_complete_train, y_complete_train, color="green", label="interpolated pts", alpha=0.5)
    # else:
    plt.scatter(x_train, y_train, marker="*", color="blue", label="used pts") #, color="lightgreen"
    
    if y_ground_truth != []: 
        # xs, ys = zip(*sorted(zip(x_train.data.numpy(), y_ground_truth)))
        xs, ys = zip(*sorted(zip(x, y_ground_truth)))
        plt.plot(xs, ys, color="grey", label="ground truth")
    plt.grid()
    plt.xlabel("input")
    plt.ylabel("output")
    plt.legend()
    _name = path+name if path else name
    plt.savefig(_name, bbox_inches='tight')
    plt.close()


def read_from_file_plots(filename_list=None, plot_name="", key = 'loss_train'):
    name = ["moxco", "vanilla"]
    for idx,  file_name in enumerate(filename_list):
        with open(file_name, 'r') as f:
            nested_dict = json.load(f)
    
        total_items = len(nested_dict)
        nested_dict = convert_lists_to_np_arrays(nested_dict)

        item_list = []
        for i in range(total_items):
            item_list.append(nested_dict[str(i)][key][-1])
        
        plt.plot(list(range(total_items)), item_list, label=name[idx])
    plt.legend()
    plt.grid()
    plt.ylabel(key)
    plt.xlabel("#runs")
    plt.savefig(plot_name)


def compare_two_curves_per_algo(filename_list=None, plot_name="", key1 = 'loss_train', key2="loss_val", file3=False, key3=None):
    name = ["moxco", "vanilla", "optimal"]
    color_map = ["blue", "green", "black"]
    for idx,  file_name in enumerate(filename_list):
        with open(file_name, 'r') as f:
            nested_dict = json.load(f)
    
        total_items = len(nested_dict)
        nested_dict = convert_lists_to_np_arrays(nested_dict)

        item_list, item_list_two, item_list_three = [], [], [] if file3 == True else None
        for i in range(total_items):
            item_list.append(nested_dict[str(i)][key1][-1])
            item_list_two.append(nested_dict[str(i)][key2][-1])
            

        plt.plot(list(range(total_items)), item_list, label=f"{name[idx]}_{key1}", linestyle="--", color=color_map[idx])
        plt.plot(list(range(total_items)), item_list_two, label=f"{name[idx]}_{key2}", color=color_map[idx])

    plt.xlabel("run number")
    plt.legend()
    plt.grid()
    plt.savefig(plot_name)


def learning_curve_plot_per_run(file_name=None, plot_name="", type="moxco"):
    results_dir = RES_DIR + timestamp()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Results directory {results_dir} made ...")

    if type== "moxco":
        arg_keys=['loss_train', 'loss_val', 'mse_true', 'mse_true_test', 'moxco_start_epoch', 'ran_where']
    else:
        arg_keys=['loss_train', 'loss_val', 'mse_true', 'mse_true_test', 'ran_where']
   
    with open(file_name, 'r') as f:
        nested_dict = json.load(f)
    
    total_items = len(nested_dict)
    nested_dict = convert_lists_to_np_arrays(nested_dict)

    
    for i in range(total_items):
        print("plotting plot number : ", i)
        key_list = []
        for key in arg_keys:
            key_list.append(nested_dict[str(i)][key])
            print(i, key, ":", nested_dict[str(i)][key])
        if type == "moxco":
            loss_train, loss_val, mse_true, mse_true_test, moxco_start_epoch, ran_where  = key_list
            mse_pred_true_log(loss_train, loss_val, mse_true, mse_true_test, moxco_start_epoch, ran_where=ran_where+"_"+str(i), path=results_dir+"/")
        else:
            loss_train, loss_val, mse_true, mse_true_test, ran_where  = key_list
            mse_pred_true_log(loss_train, loss_val, mse_true, mse_true_test, 0, ran_where=ran_where+"_"+str(i), path=results_dir+"/")


def fitted_curve_plot_per_run(file_name=None, type="moxco"):
    results_dir = RES_DIR + timestamp()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Results directory {results_dir} made ...")

    arg_keys=['x_complete_train', 'y_complete_train', 'x_train', 'y_train', 'preds', 'x', 'y_true']

    with open(file_name, 'r') as f:
        nested_dict = json.load(f)
    
    total_items = len(nested_dict)
    nested_dict = convert_lists_to_np_arrays(nested_dict)

    
    for i in range(total_items):
        print("plotting plot number : ", i)
        key_list = []
        for key in arg_keys:
            key_list.append(nested_dict[str(i)][key])
        x_complete_train, y_complete_train, x_train, y_train, preds, x, y_true  = key_list

        plot_name = f"2nn_{type}_edited_{str(i)}.png"
        visualize_boundaries_latest(x_complete_train, y_complete_train, x_train, y_train, preds, x, y_ground_truth = y_true, 
                                    name=plot_name, interpolate_data=True, path=results_dir+"/")


def density_plots(file_name=None, key='loss_train', hist_bin=10):
    with open(file_name, 'r') as f:
        nested_dict = json.load(f)
    
    total_items = len(nested_dict)
    nested_dict = convert_lists_to_np_arrays(nested_dict)

    data  = []
    for i in range(total_items):
        data.append(nested_dict[str(i)][key][-1])
    
    # plt.hist(data, bins=hist_bin, density=True, alpha=0.2, color='blue', label=f"{hist_bin}")
    
    df = pd.DataFrame(data, columns=["value"])
    sns.kdeplot(data=df, x="value", clip=(-0.1, 1), color='seagreen')
    
    plt.grid()
    plt.title(key)
    plt.xlabel(f"{key} values")
    name = "density.png"
    plt.savefig(name)


def grad_norm_after_stopping(file_name=None, key='grad_norm'):
    """ plot code for plotting grad norm after stopping in moxco """

    results_dir = RES_DIR + timestamp()
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Results directory {results_dir} made ...")

    with open(file_name, 'r') as f:
        nested_dict = json.load(f)
    
    total_items = len(nested_dict)
    nested_dict = convert_lists_to_np_arrays(nested_dict)

    base_key = "moxco_start_epoch"
    for i in range(total_items):
        start_idx = nested_dict[str(i)][base_key]
        print("plotting plot number : ", i, " mocxo stop: ", start_idx)
        if start_idx  > 0:
            grad_norm_sublist = nested_dict[str(i)][key][start_idx:]

            plt.figure()
            plt.plot(list(range(len(grad_norm_sublist))), grad_norm_sublist)
            plt.grid()
            plt.xlabel("epochs")
            plt.title(base_key)

            name = f"2nn_moxco_edited_{str(i)}.png"
            _name = results_dir+"/"+name if results_dir else name
            plt.savefig(_name, bbox_inches='tight')
            plt.close()

       
def plot_cdf(filename_list=None, key1="loss_train", plot_name="cdf.png"):
    """
    code for single plot which shows CDF of final key values 
    """
    plt.figure(figsize=(10,6))
    name = ["moxco", "vanilla"]
    markerstyle_map = ["o", "x"]
    color_map = ["blue", "green"]
    for idx,  file_name in enumerate(filename_list):
        with open(file_name, 'r') as f:
            nested_dict = json.load(f)
    
        total_items = len(nested_dict)
        nested_dict = convert_lists_to_np_arrays(nested_dict)

        item_list = []
        for i in range(total_items):
            item_list.append(nested_dict[str(i)][key1][-1])

        item_list = np.array(item_list)
        sorted_indices = np.argsort(item_list)
        sorted_mse_values = item_list[sorted_indices]

        mse_pdf = sorted_mse_values / np.array(sorted_mse_values).sum()
        mse_cdf = np.cumsum(mse_pdf)

        plt.plot(sorted_mse_values, mse_cdf, label=f"{name[idx]}_{key1}", color=color_map[idx], marker=markerstyle_map[idx], markevery=5, markersize=8)
    
    
    plt.grid()
    plt.xlabel(f"{key1}")

    plt.legend()
    plt.ylabel("Cumulative probability")
    plt.savefig(plot_name)
    plt.close()


# plot_cdf(filename_list=[filename_1, filename_2], key1="mse_true")
