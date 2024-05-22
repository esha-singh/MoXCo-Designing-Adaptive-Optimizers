import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import collections

# from toy_helper_2d import objective_function_edited, objective_function_plot_xy_independent, xy_meshgrid
from toy_helper_2d_es import get_eigen, objective_function_edited, objective_function_plot_xy_independent, xy_meshgrid
# from loss_function_graphs import beale_function, rosenbrock_function, goldstein_price_function, ackley_function

seed = 369
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(mode=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def hypergradient(hyper_lr, sub_state):
    ht = sub_state['grad']*sub_state['ut_grad']
    alpha_t = sub_state['alpha_t'] - hyper_lr*ht
    ut = -alpha_t*sub_state['grad']
    ut_grad = -sub_state['grad']
    return ut, ut_grad, alpha_t


def train_model_adam_adamom(alpha, beta, initial_guess, lr, nb_epochs = 12000, func="beale", temperature=0.01, weight_decay=0.001, mode=None):
    ''' capturable=False, differentiable=False: these two flags means no torch no_grad and interfere with calculations.'''
    # initialize a model
    WARM_START = False
    model_named_parameters = {}
    model_named_parameters["l0"]=  torch.tensor(initial_guess[0],  dtype=torch.float32, requires_grad=True, device=device)
    model_named_parameters["l1"] = torch.tensor(initial_guess[1],  dtype=torch.float32, requires_grad=True, device=device)
    prev = {}
    
    for n, p in model_named_parameters.items():
        prev[n] = p.clone()
   
    beta_history, list_gradient, list_corrected_gradient, losses = [], [], [], []

    eps = 1e-8
    x_prev, m_t, v_t = 0, {}, {}
    beta_1, beta_2 = 0.9, 0.999
    hyper_lr = 0.001
    # recent addition
    once = True
    after_mode = None
    START = False
    hessian_vector = True
    begin_lr = lr
    edge_of_stability_bound = 2*(1+alpha)/((1 + 2*beta)*(begin_lr))
    print("EOS", edge_of_stability_bound)
    moxco_start_epoch = []

    seed = 4467#random.randint(0,6553)
    np.random.seed(seed) # 1760
    print("SEED:", seed)
    for j in range(0, nb_epochs):
        if START == True:
            random_beta = beta
        else:
            np.random.seed(seed)
            random_beta = float(np.random.uniform(0.7,0.99,1))
        print("RANDOM BETA: ", random_beta)

        adict, xt_dict = {}, {}
        for n, p in model_named_parameters.items():
            prev_iter_wts = prev[n].data.clone()
            
            x_t = p.data.clone() + (alpha * (p.data.clone() - prev_iter_wts))
            xt_dict[n] = x_t.clone()

            y_t = p.data.clone() + (beta * (p.data.clone() - prev_iter_wts))
            adict[n] = p.clone()
            p.data.copy_(y_t)
            
        prev = adict
        xt_dict_main = xt_dict

        # forward pass
        beta_hat = [x for _,x in model_named_parameters.items()]
        loss = objective_function_edited(beta_hat, FLAG=func)

        for n, p in model_named_parameters.items():
            if p.requires_grad and p.grad is not None:
                p.grad.zero_()


        with torch.no_grad():
            b_current = np.array([k.item() for _,k in model_named_parameters.items()])
            b_current_ordered = np.array((b_current[0], b_current[1])) #reorder (bias, weight) #1, 0
        beta_history.append(b_current_ordered)
        
        
        loss.backward()

        # Store the gradient
        with torch.no_grad():
            grad = np.zeros(2)
            for index_p, (_,p) in enumerate(model_named_parameters.items()):
                grad[index_p] = p.grad.detach().data
            grad_ordered = np.array((grad[1], grad[0])) #reorder (bias, weight)
            list_gradient.append(grad_ordered)
        
        with torch.no_grad():
            grad = np.zeros(2)
            for index_p, (_,p) in enumerate(model_named_parameters.items()):
                grad[index_p] = p.grad.detach().data
            #reorder (bias, weight)
            grad_ordered_1 = np.array((grad[0], grad[1]))
            
            # min_grad = min(min_grad, np.linalg.norm(grad_ordered_1))
            list_corrected_gradient.append(grad_ordered_1)
        
        losses.append(loss.item())
        # STEP: adam equation with inertial momentum
        if after_mode !="adam":
            with torch.no_grad():
                for i, (n, p) in enumerate(model_named_parameters.items()):
                    x_t = xt_dict_main[n]
                    if j == 0: 
                        m_t[n] = torch.zeros_like(p.data)
                        v_t[n] = torch.zeros_like(p.data)
                    if p.requires_grad and p.grad is not None:
                        grad = p.grad.data

                    if mode=="adamW":
                        p.mul_(1 - lr * weight_decay)
                    
                    m_t[n].lerp_(grad, 1.0 - beta_1)
                    v_t[n].mul_(beta_2).addcmul_(grad, grad, value=1.0-beta_2)
                    
                    bias_correction1 = 1.0 - beta_1 ** (i+1)
                    bias_correction2 = 1.0 - beta_2 ** (i+1)
                    step_size = lr / bias_correction1
            
                    bias_correction2_sqrt = math.sqrt(bias_correction2)
                    denom = (v_t[n].sqrt() / bias_correction2_sqrt).add_(eps)

                    # p.addcdiv_(m_t[n], denom, value=-step_size)
                    param_update = x_t.data.clone() - ((step_size * m_t[n]) / denom)
                    p.data.copy_(param_update)
        else:
            with torch.no_grad():
                for i, (n, p) in enumerate(model_named_parameters.items()):
                    print("STATE:", state, p.grad)
                    x_t = xt_dict_main[n]
                    state[n]['grad'] = p.grad.clone()
                    ut, ut_grad, alpha_t = hypergradient(hyper_lr, state[n])
                    state[n]['alpha_t'] = alpha_t
                    state[n]['ut_grad'] = ut_grad

                    if p.requires_grad and p.grad is not None:
                        grad = p.grad.data
                    
                    m_t[n].lerp_(grad, 1.0 - beta_1)
                    v_t[n].mul_(beta_2).addcmul_(grad, grad, value=1.0-beta_2)

                    if p.requires_grad and p.grad is not None:
                        param_update = x_t.data + ut
                        p.data.copy_(param_update)



        if hessian_vector:
                model_params = dict(model_named_parameters)
                largest_eigenval, _ = get_eigen(model_params, epoch = j, func=func, maxIter = 3, mode="estimate")

                print("---- LARGEST EIGEN ----", "Exact: ", largest_eigenval)
                largest_eigenval_term = largest_eigenval#/(edge_of_stability_bound)
                
        else:
                largest_eigenval_term = 0

        # momemtum warm restart 
        if WARM_START == False:
            with torch.no_grad():
                grad_list = np.zeros(2)
                grad_ord, grad_norm = None, 0
                for index_p, (_,p) in enumerate(model_named_parameters.items()):
                    grad_list[index_p] = p.grad.data.cpu().detach().numpy()
                grad_ord = np.array((grad_list[0], grad_list[1]))

            grad_norm = np.linalg.norm(grad_ord)**2
            # f_x = float(loss.item()) +  grad_norm
            # p_window = np.exp(-f_x*temperature)
            middle = abs(largest_eigenval_term - 1.5*edge_of_stability_bound)/(1.5*edge_of_stability_bound)

            f_x = loss.item() + np.linalg.norm(grad_ord)**2 + middle #largest_eigenval_term#*
            p_window = np.exp(-f_x*0.15)#p_var = np.exp(-variance*0.025)# prob.append(p_window)
            print("*"*30, f"window prob: {p_window}  f_x: {f_x}  grad_norm {grad_norm}   Epoch: {j}  betas: {beta_1, beta_2}, middle: {middle}", "*"*30)
            
            if p_window > 0.80:
                WARM_START = True
                START = True
                moxco_start_epoch.append(j)
                
                print("---------------------------------------WARM START-----------------------------------------")
                alpha = 0.1#0.5
                beta = 0.1#0.5

                after_mode = "adam"
                if once == True:
                    state = collections.defaultdict(dict)
                    for n, p in model_named_parameters.items():
                        state[n]['ut_grad'] = torch.tensor(0.)
                        state[n]['alpha_t'] = hyper_lr
                        if p.requires_grad and p.grad is not None:
                            grad = p.grad.data

                        if mode=="adamW":
                            p.mul_(1 - lr * weight_decay)
                        
                        m_t[n].lerp_(grad, 1.0 - beta_1)
                        v_t[n].mul_(beta_2).addcmul_(grad, grad, value=1.0-beta_2)
                        
                        bias_correction1 = 1.0 - beta_1 ** (i+1)
                        bias_correction2 = 1.0 - beta_2 ** (i+1)
                        step_size = lr / bias_correction1
                
                        bias_correction2_sqrt = math.sqrt(bias_correction2)
                        denom = (v_t[n].sqrt() / bias_correction2_sqrt).add_(eps)

                        state[n]['grad'] =  m_t[n] / denom
                    print("-----------------------------------<<<<HYPER>>>>--------------------------------------")
                once = False


    
        # losses.append(loss.item())
        print(f"loss:  {loss.item()}  lr:  {lr}  Epoch: {j}  betas: {beta_1, beta_2}  alpha: {alpha}")
       
        moxco_start_ = j if moxco_start_epoch == [] else moxco_start_epoch[0]
        # convergence criteria
        print("Diff:", abs(x_prev - loss.item()))
        # if abs(x_prev - loss.item()) <= 2e-4:#2e-5: #2e-4
        if abs(x_prev - loss.item()) <= 1e-4 and np.linalg.norm(grad_ord) <= 1e-3:
            return beta_history, list_gradient, losses
        x_prev = loss.item()
        
    return beta_history, list_gradient, losses, moxco_start_



def inertial_prox_naive(theta_i, lr, alpha=0.9, beta=0.9, epochs=100, func="beale", temperature=0.01):
    """ theta_i: slope, intercept: b0, b1. It is linked to NN module. Slope is param1 & intercept param2
    1. inside train_model everything is reordered to bias, weight: intecept, slope.
    """

    history, list_gradient, cost, moxco_start_epoch = train_model_adam_adamom(alpha, beta, theta_i, lr, epochs, func=func, temperature=temperature, mode="adamW") # last config : 0.99, 0.99 #a=0.99, b =0.9#
    theta = history[-1] 
    print("Parameter name: ", "Intercept", "Slope")
    print("Final point: {:.2f}, {:.2f} {:d}".format(theta[0], theta[1], 50))
    # print("Least Squares: {:.2f}, {:.2f}".format(intercept, slope))
    return history, theta, cost, list_gradient, moxco_start_epoch


def plot_contour(min_val, max_val, filename, func=None):
    X_grid, Y_grid, Z_grid = xy_meshgrid(min_val, max_val)

    for i in range(len(X_grid)):
        for j in range(len(Y_grid)):
            beta_local = np.array([X_grid[i][j], Y_grid[i][j]])
            function_value = objective_function_plot_xy_independent(beta_local, test_func=func)       
            Z_grid[i, j] = function_value

    
    fig, ax = plt.subplots()
    # beale & rosenbrock
    cp = plt.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap="RdYlBu_r") # beale cmap="RdYlBu_r", "terrain"
    for line in cp.collections:
        line.set_linewidth(2.5)
    
    # goldstein  & ackley
    # cp = plt.contour(X_grid, Y_grid, Z_grid, levels=np.logspace(0, 1, 45), norm=LogNorm(), cmap="RdYlBu_r") # beale cmap="RdYlBu_r", "terrain"

    # plt.clabel(cp, inline=2, fontsize=10)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.savefig(filename)
    return X_grid, Y_grid, filename, ax
    

def plot_gradient_flow(min_val, max_val, lr, alpha=0.9, beta=0.9, epochs=100, temperature=0.01, func="beale", file_name="2d_beale_gradient.png"):
    """ plot for one point first"""
    # min_val, max_val, delta_grid = -4.5,4.5, 0.02
    # filename = plot_gradient_contour(min_val, max_val, delta_grid)
    X_grid, Y_grid, filename, ax = plot_contour(min_val, max_val, file_name, func)

    # take one point for which to plot gradients for.
    # 0, 1 saddle point for beale
    # list_of_points = [(0.3, 0.1)]#[(-0, 3.1)]#[(0.4,-1), (0.23, 0.9),(0.5, -0.5),(2,0)]#(0.5, -0.5),(0.23, 0.9), [(-3, 1.2)]#[(0, -4)]#[(-2, 1.3)]#[(0, -3)]#[(0.5, 2)]#[(0.5, 2)]#[(-1,-2), (2,2), (0.4, -1), (0.23, 0.9), (-3, 3)] # add -2, 1
    if func == "rosenbrock":
        list_of_points = [(3, 4.8)]#[(-3, 2)] (-1.8, 2)1.5, 0.5)
    elif func == "beale":
        list_of_points = [(-2.5, 1.2)]#[(-2.5, 1.2)]#[(-2.5, 1.2)]#[(-4.3, 1.3)]#[(-0.5,1.5)]#[(0,1)]#[(-1,0)]#[(-0.5,1.5)]#[(-4.3, 1.3)] #[(-0.5,1.5)]#[(-4.3, 1.3)] #[(-3, 1.2)]
    elif func == "goldstein-price":
        list_of_points = [(-1,3.5)]#[(0,0)]#[(0,0)]#[(-1,3.5)]#[(0,3.5)]#[(0.1,-0.1)]#[(1, 0.7)]#[(1, 1.5)]#[(-2, 2)]
    elif func == "ackley":
        list_of_points = [(-2,-2)]#[(3,0)]#[(-2,3)]#[(-1,-2)]#[(-1.5,-3.5)]#[(-1,3.3)]
    color = ["red", "pink", "orange"]
    for xcoord, ycoord in list_of_points:
        # xcoord, ycoord = -1, -2
        print("*"*89)
        print("point taken ....", xcoord, ycoord)
        history, _, cost, gradient_list = inertial_prox_naive([xcoord, ycoord], lr, alpha=alpha, beta=beta, epochs=epochs, func=func, temperature=temperature)
        loop_iter = 0
        print("No. of gradient points:", len(history))
        for (i, (beta_i, grad_i)) in enumerate(zip(history, gradient_list)):
            if i%1 == 0:
                print("Epoch:", i, "beta_0: ",[beta_i[0]], "beta_1: " , [beta_i[1]], "beta norm: ", np.linalg.norm(beta_i), "grad norm: ", np.linalg.norm(grad_i), "cost: ", cost[i])

                plt.scatter([beta_i[0]], [beta_i[1]], s=100, c='red', marker="x", linewidth=2.0, label=str(loop_iter)) #mediumpurple
                #     if i > 95:
                #         plt.scatter([beta_i[0]], [beta_i[1]], s=100, c="red", marker=".", linewidth=2.0, label=str(loop_iter)) #mediumpurple
                #     else:
                        # plt.scatter([beta_i[0]], [beta_i[1]], s=100, c="orange", marker=".", linewidth=2.0, label=str(loop_iter)) #mediumpurple
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
        plt.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c="black", linewidth=1)
    elif func== "beale":
        plt.plot(3,0.5, 'b*', markersize=18)
    elif func == "goldstein-price":
        plt.plot(0,-1, 'b*', markersize=18)
    elif func == "ackley":
        plt.plot(0,0, 'b*', markersize=18)
        plt.plot([a for a, _ in history], [b for _, b in history], linestyle="--", c="black", linewidth=1)
    plt.savefig(filename)
    plt.close()


#------------ BEALE ------------
min_val, max_val = -4.5, 4.5
# old
# lr, alpha, beta, epochs, temperature = 0.01, 0.9, 0.9, 1200, 0.01 #12000

# lr, alpha, beta, epochs, temperature = 0.005, 0.96, 0.4, 1200, 0.15
# plot_gradient_flow(min_val, max_val, lr, alpha=alpha, beta=beta, epochs=epochs, temperature=temperature,  func="beale", file_name="2d_beale_gradient.png")
