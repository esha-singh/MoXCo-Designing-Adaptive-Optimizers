import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, jacobian, tree_map

 
PATH = os.path.abspath(your_path)
sys.path.append(PATH)
from HessianFlow import hessianflow as hf
# old 
# COLORS = ["orange", "blue", "red", "black", "green", "seagreen", "pruple", "pink"]
COLORS = ["orange", "blue", "red", "seagreen", "green", "seagreen", "pruple", "pink"]


def convert_lists_to_np_arrays(data):
    if isinstance(data, dict):
        return {k: convert_lists_to_np_arrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return np.array(data)
    return data

def convert_np_to_lists(data):
    if isinstance(data, dict):
        return {k: convert_np_to_lists(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Converts numpy array to list
    elif isinstance(data, np.float32):
            return float(data)
    elif isinstance(data, list):
        return [convert_np_to_lists(item) for item in data]  # Ensure all list items are also processed
    return data  # Return the item itself if it's neither a dict nor a numpy array


class Logging():
    def __init__(self, path="", per_run_epochs=10000, save_file_name="all_runs_results.json"):
        self.path = path
        self.per_run_epochs = per_run_epochs
        self.save_file_name = save_file_name
    
    def save(self, nested_dict):
        with open(self.save_file_name,"w") as f:
            converted_data = convert_np_to_lists(nested_dict)
            json.dump(converted_data, f,  indent=4)
    
    def plot_curves(self, alist, blist, c="black", plot_label=""):
        plt.plot(alist, blist, color=c, label=plot_label)
        plt.legend()
        plt.grid()
        plt.xlabel("# trials")
        plt.ylabel("loss_train")




def visualize_boundaries(x_test, y_test, preds, y_ground_truth = None, name=None, color=None, path=None):
    plt.scatter(x_test,preds.data.numpy(), color="black")
    plt.scatter(x_test,y_test, color="green")
    if y_ground_truth:
        xs, ys = zip(*sorted(zip(x_test.data.numpy(), y_ground_truth)))
        plt.plot(xs, ys, color="grey")
    plt.grid()
    plt.xlabel("input")
    plt.ylabel("output")
    _name = path+name if path else name
    plt.savefig(_name, bbox_inches='tight')
    plt.close()


def visualize_boundaries_latest(x_complete_train, y_complete_train, x_train, y_train, preds, x, 
                                y_ground_truth = None, nn_fit_line = True, 
                                interpolate_data=True, name=None, path=None):
    plt.figure(figsize=(7, 4))
    if nn_fit_line:
        xs, ys = zip(*sorted(zip(x_train.data.numpy(), preds.data.numpy())))
        plt.plot(xs, ys, color="black", label="fitted")
    else:
        plt.scatter(x_train, preds.data.numpy(), color="black")

    if interpolate_data:
        plt.scatter(x_complete_train, y_complete_train, color="green", label="interpolated pts", alpha=0.5)
    # else:
    plt.scatter(x_train, y_train, marker="*", color="blue", label="used pts") #, color="lightgreen"
    
    if y_ground_truth: 
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


def loss_plots(loss_values, ran_where=None):
    # plt.yscale('log')
    plt.plot(list(range(len(loss_values))), loss_values)
    plt.ylim(0, 5)
    plt.grid()
    
    plt.savefig("loss_plot_"+ ran_where + ".png")
    plt.close()

def train_val_plots(train_loss, val_loss, ran_where="vanilla", path=None):
    # plt.yscale('log')
    plt.plot(list(range(len(train_loss))), train_loss, label="train")
    plt.plot(list(range(len(val_loss))), val_loss, label="val")
    plt.ylim(0, 3)
    plt.grid()
    plt.legend()
    name = "train_val_plot_" + ran_where + ".png"
    _name = path+name if path else name
    plt.savefig(_name)
    plt.close()


def mse_pred_true(mse_pred, mse_true, ran_where="vanilla", path=None):
    plt.plot(list(range(len(mse_pred))), mse_pred, label="mse pred train")
    # plt.plot(list(range(len(mse_true))), mse_true, label="mse groud truth train")
    plt.ylim(0, 5)
    plt.grid()
    plt.legend()
    name = "mse_pred_true_"+ ran_where + ".png"
    _name = path+name if path else name
    plt.savefig(_name)
    plt.close()


def mse_pred_true_all(train_loss, val_loss, mse_true, mse_true_test, moxco_start_epoch, ran_where="vanilla", path=None):
    # plt.plot(list(range(len(mse_pred))), mse_pred, label="mse pred train")
    # plt.yscale('log')
    # plt.xscale('log')
    plt.plot(list(range(len(train_loss))), train_loss, label="train")
    plt.plot(list(range(len(val_loss))), val_loss, label="test")
    plt.plot(list(range(len(mse_true))), mse_true, label="mse groud truth train")
    # plt.plot(list(range(len(mse_true_test))), mse_true_test, label="mse groud truth test")
    if moxco_start_epoch > 0:
        plt.axvline(x=moxco_start_epoch, linestyle="--")

    plt.ylim(0, 2)
    plt.grid()
    plt.legend()
    name = "mse_pred_true_all_"+ ran_where + ".png"
    _name = path+name if path else name
    plt.savefig(_name)
    plt.close()


def mse_pred_true_log(train_loss, val_loss, mse_true, mse_true_test, moxco_start_epoch, ran_where="vanilla", path=None):
    # plt.plot(list(range(len(mse_pred))), mse_pred, label="mse pred train")
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(list(range(len(train_loss))), train_loss, label="train")
    plt.plot(list(range(len(val_loss))), val_loss, label="test")
    plt.plot(list(range(len(mse_true))), mse_true, label="mse groud truth train")
    # plt.plot(list(range(len(mse_true_test))), mse_true_test, label="mse groud truth test")
    if moxco_start_epoch > 0:
        plt.axvline(x=moxco_start_epoch, linestyle="--")

    # plt.ylim(0, 2)
    plt.grid()
    plt.legend()
    name = "mse_pred_true_log_"+ ran_where + ".png"
    _name = path+name if path else name
    plt.savefig(_name)
    plt.close()


def eigenvalues(model, criterion, inputs, targets):
    eigenvalue, eigenvec = hf.get_eigen(model, inputs, targets, criterion, cuda = False, maxIter = 3, tol = 1e-3)
    print('\nCurrent Eigenvalue based on Test Dataset: %0.2f' % eigenvalue)
    print("Eigenvalues..", eigenvalue, len(eigenvec))
    return eigenvalue

def pred(x, params):
    relu = torch.nn.ReLU()
    W1, W2, b1, b2 = params
    z1 = torch.matmul(x, W1) + b1  
    a1 = relu(z1)
    scale_factor = torch.sqrt(torch.tensor(2. / 5))
    z2 = torch.matmul(a1, W2) * scale_factor + b2
    return z2

from jax.nn import relu
def get_eigen_1d(x, y, o, model, hidden_size=1):
    """
    1d: compute the top eigenvalues of model parameters and 
    the corresponding eigenvectors.
    """
    x = torch.tensor(x)
    y = torch.tensor(y)
    def loss(y_pred, y):
        return torch.mean(torch.square(y_pred - y))

    def pytorch_hessian(x, func, y_pred, y):    
        m = func(y_pred, y)
        grads = torch.autograd.grad(m, x, create_graph=True)
        hessian = np.zeros((hidden_size, hidden_size))
        # Compute the Hessian
        for idx, grad in enumerate(grads):
            grad2 = torch.autograd.grad(grad, x, grad_outputs=torch.ones_like(grad), retain_graph=True)
            for j in range(len(grad2)):
                hessian[idx, j] = np.linalg.norm(grad2[j].numpy())
        return hessian

    params = [torch.tensor(p, requires_grad=True) for _,p in model.items()]
    y_pred = pred(x, params)
    out = pytorch_hessian(params, loss, y_pred, y)
    
    out = np.array(out)
    
    eigenvalues_, _ = np.linalg.eig(out)
    eigenvalue = max(eigenvalues_)
    print('\nCurrent Eigenvalue based on Test Dataset: %0.2f' % eigenvalue)
    return eigenvalue

def get_eigen(x, y, o, model, hidden_size=1):
    """
    1d: compute the top eigenvalues of model parameters and 
    the corresponding eigenvectors.
    """
    # x = torch.tensor(x)
    # y = torch.tensor(y)
    def loss(y_pred, y):
        return torch.mean(torch.square(y_pred - y))

    maxIter = 3
    params = [torch.tensor(p, requires_grad=True) for _,p in model.named_parameters()] #model.items()
    y_pred = pred(x, params)
    ff = loss(y_pred, y)
    grad_list = torch.autograd.grad(ff, params, create_graph=True) # loss.backward(create_graph = True)

    eigenvalue = None
    v = [torch.randn(p.size()) for p in params]
    v = hf.utils.normalization(v)
    for i in range(maxIter):
        Hv = torch.autograd.grad(grad_list, params, grad_outputs = v, retain_graph = True)
        eigenvalue_tmp =  hf.utils.group_product(Hv, v).item()
        a, b, c, d = Hv[0], Hv[1], Hv[2], Hv[3]
        a, c, d = a.view(hidden_size, 1), c.view(hidden_size, 1), d.view(1, 1).expand(hidden_size, 1) 
        concatenated_tensor = torch.cat((a,b,c,d), dim=1)
        norm = torch.norm(concatenated_tensor)
        v = [i/norm.item() for i in Hv]
        # print("norm of v:", v, eigenvalue_tmp, Hv)
    
    print('\nCurrent Eigenvalue based on Test Dataset: %0.2f' % eigenvalue_tmp)
    return eigenvalue_tmp

def get_eigne_jax(x, y, o, model, hidden_size=1):
    # Define the loss function
    def pred_jax(x, params):
        W1, W2, b1, b2 = params["w1"], params["w2"], params["b1"], params["b2"]
        z1 = jnp.dot(x, W1) + b1  # Use jnp.dot for matrix multiplication
        a1 = relu(z1)  # JAX's relu function
        scale_factor = jnp.sqrt(jnp.array(2. / 5))
        z2 = jnp.dot(a1, W2) * scale_factor + b2
        return z2

    def loss(params, y_pred, y):
    # Assuming that we need to adjust y_pred to match dimensionality requirements
        # Here we simply ensure y_pred is a vector that matches the inner dimension of w1.
        # Since w1 is [1, 5], we need y_pred to be [5,].
        if y_pred.size != params['w1'].shape[1]:
            # This is a placeholder; in real scenarios, adjust y_pred or model architecture accordingly.
            y_pred_adjusted = jnp.ones(params['w1'].shape[1])  # Creating a dummy y_pred for dimensionality compatibility
        else:
            y_pred_adjusted = y_pred

        prediction = params['w1'] @ y_pred_adjusted.reshape(-1, 1)  # Ensure y_pred is column vector
        prediction += jnp.sum(params['w2'])  # Adding scalar sum of w2
        prediction += jnp.dot(params['b1'], y_pred_adjusted)  # Dot product
        prediction += params['b2']  # Adding scalar b2
        return jnp.mean(jnp.square(prediction.squeeze() - y))  # Ensure scalar output


    # Define a function to compute the Hessian matrix of the loss with respect to parameters
    def hessian(params, y_pred, y):
        # Return the jacobian of the gradient (Hessian)
        return jacobian(grad(loss, argnums=0))(params, y_pred, y)

    # Compile for performance using JIT
    compiled_hessian = jit(hessian)


    # params = [torch.tensor(p, requires_grad=True) for _,p in model.items()]
    
    
    params = [jnp.array(p) for _,p in model.items()]
    params = dict(zip(["w1", "w2", "b1", "b2"], params))

    y = jnp.array(y.data)
    y_pred = pred_jax(x, params)
    
    # print(y_pred)

    # Compute the Hessian matrix
    hessian_matrix = compiled_hessian(params, y_pred, y)
    # print(hessian_matrix)

    # Example structure, simplified dimensions:
    params_order = ['b1', 'b2', 'w1', 'w2']
    params_sizes = {
        'b1': 5,  # Assuming each block matrix for 'b1' is 5x5
        'b2': 1,  # Assuming each block matrix for 'b2' is 1x1
        'w1': 5,  # Assuming each block matrix for 'w1' is 5x5
        'w2': 5   # Assuming each block matrix for 'w2' is 5x5
    }

    # Initialize a large Hessian matrix:
    total_size = sum(params_sizes.values())
    full_hessian = np.zeros((total_size, total_size))

    # We need to fill this matrix with your data:
    start_i = 0
    for i, param_i in enumerate(params_order):
        end_i = start_i + params_sizes[param_i]
        start_j = 0
        for j, param_j in enumerate(params_order):
            end_j = start_j + params_sizes[param_j]
            # Insert the block matrix:
            block = hessian_matrix[param_i][param_j]  # Fetch the block
            full_hessian[start_i:end_i, start_j:end_j] = block.reshape(params_sizes[param_i], params_sizes[param_j])
            start_j = end_j
        start_i = end_i

    # Now `full_hessian` is filled with the full Hessian matrix
    eigenvalues = np.linalg.eigvals(full_hessian)
    print("Eigenvalues of the Full Hessian Matrix:", max(eigenvalues))
    return max(eigenvalues)

def get_eigen_hf(x, y, o, model, hidden_size=1):
    # issue of model.eval call
    x = torch.tensor(x)
    y = torch.tensor(y)
    def loss(y_pred, y):
        return torch.mean(torch.square(y_pred - y))
    def get_eigen(model, inputs, targets, criterion, cuda = True, maxIter = 50, tol = 1e-3):
        """
        compute the top eigenvalues of model parameters and 
        the corresponding eigenvectors.
        """

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward(create_graph = True)

        params, gradsH = get_params_grad(model)
        v = [torch.randn(p.size()).to(device) for p in params]
        v = normalization(v)

        eigenvalue = None

        for i in range(maxIter):
            model.zero_grad()
            Hv = hessian_vector_product(gradsH, params, v)
            eigenvalue_tmp = group_product(Hv, v).cpu().item()
            v = normalization(Hv)
            # concatenated_tensor = torch.stack(Hv, dim=0)
            # norm = torch.norm(concatenated_tensor)
            # v = [i/norm.item() for i in Hv]
            if eigenvalue == None:
                eigenvalue = eigenvalue_tmp
            else:
                if abs(eigenvalue-eigenvalue_tmp)/abs(eigenvalue) < tol:
                    return eigenvalue_tmp, v
                else:
                    eigenvalue = eigenvalue_tmp
    return eigenvalue, v

    def eigenvalues_test(model):
        criterion = lambda y_pred, y: loss(y_pred, y)
        eigenvalue, eigenvec = hf.get_eigen(model, x, y, criterion, cuda = False, maxIter = 3, tol = 1e-3)
        print('\nCurrent Eigenvalue based on Test Dataset: %0.2f' % eigenvalue)
        print("Eigenvalues..", eigenvalue, len(eigenvec))
        return eigenvalue
    
    ei = eigenvalues_test(model)
    return max(ei)

def get_eigen_test(x, y, y_pred, model, hidden_size=2):
    # def get_eigen(model_parameters, epoch = None, func=None, maxIter = 2, tol = 1e-6, mode = "exact"):#1e-3):
    """
    compute the top eigenvalues of model parameters and 
    the corresponding eigenvectors.
    """
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    # kk = [p.data.cpu().detach().numpy() for _,p in model.named_parameters()]
    params  = [p.data.cpu().detach() for _,p in model.named_parameters()]#[torch.tensor(item, requires_grad=True) for item in kk]
    print("--", params)
    
    loss = loss_fn(y_pred, y)
    grad_list = torch.autograd.grad(loss, params, create_graph=True) # loss.backward(create_graph = True)

    # eigenvalue = None
   
    v = [torch.randn(p.size()) for p in params]
    v = hf.utils.normalization(v)
    for i in range(2):

        # Hv = list(torch.autograd.grad([i*j for i, j in list(zip(grad_list, v))], params, retain_graph=True))
        Hv = torch.autograd.grad(grad_list, params, grad_outputs = v, retain_graph = True)
        # Hv = hessian_vector_product(loss, params, v)
        eigenvalue_tmp =  hf.utils.group_product(Hv, v).item()
        concatenated_tensor = torch.stack(Hv, dim=0)
        norm = torch.norm(concatenated_tensor)
        v = [i/norm.item() for i in Hv]
        # print("norm of v:", v, eigenvalue_tmp, Hv)
    return eigenvalue_tmp, v


def get_eigen_finite_diff(x, y, o, model, hidden_size=1):
    # https://math.stackexchange.com/questions/3254520/computing-hessian-in-python-using-finite-differences
    X = np.array(x)
    y = np.array(y)
    
    def f(params):
        W1, W2, b1, b2 = params
        relu = lambda z: np.maximum(0, z)
        z1 = X.dot(W1) + b1  
        a1 = relu(z1)
        scale_factor = 1
        scale_factor = np.sqrt(2./hidden_size)
        z2 = a1.dot(W2)*scale_factor + b2 
        mse_loss = ((z2 - y) ** 2).mean()
        return mse_loss
    
    def normalize_hessian(H):
        D = np.sqrt(np.diag(H))  # Square roots of diagonal elements
        D_inv = np.outer(D, D)  # Outer product to form denominator matrix
        return H / D_inv  # Element-wise division


    def finite_difference_hessian(f, x, h=1e-5):
        n = len(x)
        hessian = np.zeros((n, n))  # Initialize the Hessian matrix
        for i in range(n):
            for j in range(n):
                x_orig = [xi.copy() for xi in x]
                # Perturb x[i]
                x_ih = [xi.copy() for xi in x]
                x_ih[i] += h * np.ones_like(x[i])
                # Perturb x[j]
                x_jh = [xi.copy() for xi in x]
                x_jh[j] += h * np.ones_like(x[j])
                # Perturb both x[i] and x[j]
                x_ijh = [xi.copy() for xi in x]
                x_ijh[i] += h * np.ones_like(x[i])
                x_ijh[j] += h * np.ones_like(x[j])
                # Evaluate the function at the perturbed points
                f_ijh = f(x_ijh)
                f_ih = f(x_ih)
                f_jh = f(x_jh)
                f_original = f(x_orig)
                # Second derivative approximation
                hessian[i, j] = (f_ijh - f_ih - f_jh + f_original) / (h ** 2)
        hessian = normalize_hessian(hessian)
        return hessian
    
    params = [np.array(p) for _, p in model.items()]
    hessian_matrix = finite_difference_hessian(f, params)
    eigenvalues_, _ = np.linalg.eig(hessian_matrix)
    eigenvalue = max(eigenvalues_)
    return eigenvalue

def press_statistic(y_true, y_pred, xs):
    """
    https://gist.github.com/benjaminmgross/d71f161d48378d34b6970fa6d7378837
    Calculation of the `Press Statistics <https://www.otexts.org/1580>`_
    """
    res = y_pred - y_true
    hat = np.dot(xs, (np.linalg.pinv(xs)))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()

def predicted_r2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = press_statistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs
    )

    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - press / sst
 
def r2(y_true, y_pred):
    """
    Calculation of the unadjusted r-squared, goodness of fit metric
    """
    sse  = np.square( y_pred - y_true ).sum()
    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - sse/sst

def plot_small_vals(loss_vals, start_epoch, end_epoch, decimal_boost=10**6, ran_where="vanilla", path=None):
    # https://stackoverflow.com/questions/40329424/plot-very-small-values-with-matplotlib-in-jupyter
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    
    xs = list(range(start_epoch, end_epoch+1))
    print(xs)
    ys = [i*(decimal_boost) for i in loss_vals]#[1.248267, 1.248212, 1.248204, 1.248199, 1.248196, 1.248192, 1.248189, 1.248186,1.248182, 1.248179]
    print(ys)
    ax.plot(xs, ys, marker='.')
    ax.set_ylabel(r'Value [x 10^6]')
    name = "zoomed_part_after_zone" + ran_where + ".png"
    _name = path+name if path else name
    plt.savefig(_name)
    plt.close()


def visualize_ending_epochs(x, preds_train_values):
    last_epochs = list(range(50000, 45000, -1))
    preds_train_last_values = preds_train_values[-5000:]
    # plot using x, y, preds_train_last_values, last_epochs
    # plot figure with input and the preds and color code the objective value.
    

def plot_contour(model_params, filename="contour_of_last_epochs.png"):
    from get_data import generate_data
    X, Y, _ = generate_data(samples = 1000)

    def objective(params, x, y):
        x = np.array(x)
        W1, W2, b1, b2 = params
        relu = lambda z: np.maximum(0, z)
        z1 = x.dot(W1) + b1  
        a1 = relu(z1)
        scale_factor = 1
        scale_factor = np.sqrt(2./1000)
        z2 = a1.dot(W2)*scale_factor + b2 
        mse_loss = ((z2 - y) ** 2)
        return mse_loss
    
    # def objective(pred, y):
    #     mse_loss = ((pred - y) ** 2)
    #     return mse_loss
    
    X, Y = zip(*sorted(zip(X, Y)))
    X_grid, Y_grid = np.meshgrid(X, Y)
    Z = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            Z[i][j] = objective(model_params, X_grid[i][j], Y_grid[i][j])

    from matplotlib.colors import LogNorm
    fig, ax = plt.subplots()
    cp = plt.contour(X_grid, Y_grid, Z, cmap="RdYlBu_r") # beale cmap="RdYlBu_r", "terrain"
    
    for line in cp.collections:
        line.set_linewidth(1.1)
    plt.clabel(cp, inline=2, fontsize=10)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.savefig(filename)
    # plt.xlim(min_val, max_val)
    # plt.ylim(min_val, max_val)


def mse_pred_true_log_eigenval(train_loss, val_loss, mse_true, moxco_start_epoch, eigenvalues_list, EOS=0, ran_where="vanilla", path=None):
    # plt.plot(list(range(len(mse_pred))), mse_pred, label="mse pred train")
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(list(range(len(train_loss))), train_loss, label="train")
    plt.plot(list(range(len(val_loss))), val_loss, label="test")
    plt.plot(list(range(len(mse_true))), mse_true, label="mse groud truth train")
    plt.plot(list(range(len(eigenvalues_list))), eigenvalues_list, label="evolution of max eigenvals", color="red")
    # plt.plot(list(range(len(mse_true_test))), mse_true_test, label="mse groud truth test")
    if moxco_start_epoch > 0:
        plt.axvline(x=moxco_start_epoch, linestyle="--")
    
    
    plt.axvline(x=EOS, linestyle="--", color="black")

    # plt.ylim(0, 2)
    plt.grid()
    plt.legend()
    name = "mse_pred_true_log_"+ ran_where + ".png"
    _name = path+name if path else name
    plt.savefig(_name)
    plt.close()


def multiple_lists_per_plot(dicts=None, moxco_start_epoch=0, iterations=None, x_label="Iterations", title="Goodness score analysis", filename="criteria_analysis.pdf", log_scale=False):
    if dicts:
        color_map = COLORS[:len(dicts)]
        sym1, sym2, sym3  = r'$||\nabla g(\theta)||_2^2$', r'$\left|\frac{\lambda_{\max}(\nabla^2 f(\theta))}{\lambda_{\mathrm{target}}} -1\right|$', r'$|f(\theta) - f_{\mathrm{target}}|$'
        label_map = {"grad_norm": sym1, "largest_eigenvals":sym2, "suboptimal_loss":sym3}
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(15,10))

    if dicts['grad_norm']:
        tmp = dicts['grad_norm'] 
        dicts['grad_norm'] = [k**2 for k in tmp]

    if moxco_start_epoch > 0:
        plt.axvline(x=moxco_start_epoch, linestyle="--", color="black")

    for i, (key, alist) in enumerate(dicts.items()):
        # smoothening the curve
        if key == "largest_eigenvals" :#or key=="grad_norm":
            from scipy.signal import savgol_filter
            alist = [np.exp(-(ai-0.5)**2/0.01) for ai in alist]
            # alist = savgol_filter(alist, 21, 4) # 7/11,2
            alist = savgol_filter(alist, 501, 2)
        if log_scale:
            if key == "grad_norm_vanilla":
                plt.loglog(list(range(len(alist))), alist, label=f"Vanilla GD {sym1}", color=color_map[i], ls="--", linewidth=3)
            else:
                plt.loglog(list(range(len(alist))), alist, label=f"{label_map[key]}", color=color_map[i], linewidth=3)
        else:
            plt.plot(list(range(len(alist))), alist, label=f"{label_map[key]}", color=color_map[i], linewidth=3)
        

    plt.legend(fontsize=16, loc="lower right")
    plt.xlabel(x_label)
    plt.ylabel("values")
    plt.title(title)
    plt.grid(True, which="both", ls="--", color='gray', alpha=0.5)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    # plt.plot(list(range(len(dicts["largest_eigenvals"]))), dicts["largest_eigenvals"])
    # plt.savefig("op.png")

def line_plot(items=None, filename="snap.png", moxco_start_epoch=0):
    plt.figure()
    if len(items) == 2 :
        alist, blist = items.values()
        keys = list(items.keys())
        print(alist)

        from scipy.signal import savgol_filter
        # alist = savgol_filter(alist, 23, 6) 
        blist = savgol_filter(blist, 23, 6) 
        
        plt.plot(alist, blist) #markevery=list(range(15))

        if moxco_start_epoch > 0:
            plt.axvline(x=alist[moxco_start_epoch-1], linestyle="--", color="black")

        plt.xlabel(keys[0])
        plt.ylabel(keys[1])
        plt.grid()
        plt.savefig(filename, bbox_inches='tight')
        plt.close() 


# def plot_goodness_subopt(goodness_list, suboptimal_loss_list):
#     plt.plot([abs(x) for x in suboptimal_loss_list], goodness_list)
#     plt.grid()
#     plt.savefig("criterion_test.png")