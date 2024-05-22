import torch
import random
import numpy as np
import collections
from torch.autograd import Variable

# from utils import eigenvalues, plot_small_vals
from sklearn.metrics import mean_squared_error
"""
2-layer fully connected NN
"""

        
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(H, D_out, bias=False)

        self.b2 = torch.nn.Parameter(torch.zeros(D_out))
        self.H = H

        # old
        # self.w2 = torch.nn.Parameter(torßh.zeros(D_out, H)/np.sqrt(H))
        # self.b2 = torch.nn.Parameter(torch.zeros(D_out))
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary (differentiable) operations on Tensors.
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)/np.sqrt(self.H) + self.b2
        # x = torch.nn.functional.linear(x, self.w2, self.b2) #/np.sqrt(H) + self.b2
        return x
    

## takes in a module and applies the specified weight initialization
def _weights_init_normal(model, seed_):
    torch.manual_seed(seed_)
    for n, p in model.named_parameters():
        if n == "linear1.weight":
            with torch.no_grad():
                p.copy_(torch.sign(p.data.normal_(0, 1)))

        if n == "linear1.bias":
            p.data.uniform_(-0.5, 0.5)

        if n == "linear2.weight":
            p.data.normal_(0, 1)

        if n == "b2":
            w = torch.empty(p.size())
            p.data.zero_()


def predict(model, x, y):
    """ make predictions """
    outputs = model(x)
    return outputs


def train(model,loss_fn, x, y, learning_rate):
    y_pred = model(x)   # Forward spass: compute predicted y by passing x to the model.
    loss = loss_fn(y_pred, y) # Compute and print loss. 
    model.zero_grad()

    loss.backward() # Backward pass

    # Update the weights using gradient descent.
    with torch.no_grad():
        for n, param in model.named_parameters():
            # if n=='linear2.weight' or n=="b2":
                param.data -= learning_rate * param.grad
            
    return loss.item()


def vanilla_gd(train_X, train_Y, test_X, test_Y, Y_true, Y_true_test, seed=0, lr=0.1, iterations = 0, D_in=1, D_out=1, Hidden=1000):
    print("SEED: ", seed)
    
    H = Hidden
    model = TwoLayerNet(D_in, H, D_out) 

    _weights_init_normal(model, seed)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = lr

    plot_loss, loss_train, loss_val, mse_true, mse_true_test, grad_norm_list, prev_loss = [], [], [], [], [], [], 0

    #run the training
    for e in range(1, iterations+1):
        loss = 0.
        loss = train(model, loss_fn, train_X, train_Y, learning_rate)
        
        with torch.no_grad():
            grad_list = np.zeros(len([_ for _ in model.parameters()]))
            grad_ord, grad_norm = None, 0
            count = 0
            for _, p in model.named_parameters():
                tmp = p.grad.data.cpu().detach().numpy()
                grad_list[count] = np.linalg.norm(tmp)**2
                count += 1

            grad_ord = np.array(grad_list)
        grad_norm = np.sqrt(np.sum(grad_ord)) #np.linalg.norm(grad_ord)**2
        grad_norm_list.append(grad_norm)
        preds = predict(model, train_X, train_Y)
        preds_val = predict(model, test_X, test_Y)

        with torch.no_grad():
            loss_train.append(mean_squared_error(train_Y.data.numpy(), preds.data.numpy()))
            loss_val.append(mean_squared_error(test_Y.data.numpy(), preds_val.data.numpy()))
            mse_true.append(mean_squared_error(Y_true, preds.data.numpy()))
            mse_true_test.append(mean_squared_error(Y_true_test, preds_val.data.numpy()))

        plot_loss.append(loss)
        print("Epoch %02d, loss = %f, grad = %f" % (e, loss, grad_norm))

        # converging criterias
        if np.linalg.norm(grad_ord) <= 1e-5 or abs(prev_loss - loss) < 1e-5: # this was for normal local hit rate ones 

            print("converging criteria met .... ")
            return plot_loss, preds, loss_train, loss_val, preds_val, mse_true, mse_true_test,  0, seed, grad_norm_list
        
        prev_loss = loss
    return plot_loss, preds, loss_train, loss_val, preds_val, mse_true, mse_true_test,  0, seed, grad_norm_list


def hypergradient(hyper_lr, sub_state):
    ht = sub_state['grad']*sub_state['ut_grad']
    alpha_t = sub_state['alpha_t'] - hyper_lr*ht
    ut = -alpha_t*sub_state['grad']
    ut_grad = -sub_state['grad']
    return ut, ut_grad, alpha_t


def train_moxco_hypergradient(model,loss_fn, x, y, prev, learning_rate, state, hyper_lr=0, alpha=0, beta=0, hyper_gradient=False):
    y_pred = model(x)   # Forward pass: compute predicted y by passing x to the model.
    
    adict, xt_dict = {}, {}
    for n, p in model.named_parameters():
        
        prev_iter_wts = prev[n].data.clone()

        x_t = p.data.clone() + (alpha * (p.data.clone() - prev_iter_wts))
        xt_dict[n] = x_t.clone()

        y_t = p.data.clone() + (beta * (p.data.clone() - prev_iter_wts))
        adict[n] = p.clone()
        # if n=='linear2.weight':
        # if n=='linear2.weight' or n=="b2":
        p.data.copy_(y_t)

    prev = adict
    xt_dict_main = xt_dict
    
    loss = loss_fn(y_pred, y) # Compute and print loss.

    model.zero_grad()
    loss.backward() # Backward pass

    # < ------------------------------------------------ >
    # Step update
    if hyper_gradient == True:
        for n, p in model.named_parameters():
            x_t = xt_dict_main[n]
            state[n]['grad'] = p.grad.clone()
            ut, ut_grad, alpha_t = hypergradient(hyper_lr, state[n])
            state[n]['alpha_t'] = alpha_t
            print("alpha_t: ", n, torch.norm(alpha_t))
            state[n]['ut_grad'] = ut_grad
            if p.requires_grad and p.grad is not None:
                # if n=='linear2.weight':
                # if n=='linear2.weight' or n=="b2":
                    param_update = x_t.data + ut
                    p.data.copy_(param_update)
    else:
        with torch.no_grad():
            for n, param in model.named_parameters():
                x_t = xt_dict_main[n]
                if param.requires_grad and param.grad is not None:
                    # if n=='linear2.weight':
                    # if n=='linear2.weight' or n=="b2":
                        param_update = x_t.data.clone() - (learning_rate * param.grad)
                        param.data.copy_(param_update)
        
    return loss.item(), prev


def moxco_hypergradient(train_X, train_Y, test_X, test_Y, Y_true, Y_true_test, seed=0, is_hessian=True, lr=0.001, iterations = 0, D_in=1, D_out=1, 
                        Hidden=1000, eta=0.85, tau=0.2, alpha=0.6, beta=0.6, hyper_lr=1e-2):
    
    print("SEED: ", seed)
    print("Hidden: ", Hidden)
    H = Hidden 
    learning_rate = lr

    model = TwoLayerNet(D_in, H, D_out) 
    _weights_init_normal(model, seed)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    
    plot_loss, loss_train, loss_val, mse_true, mse_true_test, grad_norm_list, goodness_scores, suboptimal_loss_list, eigenvalues_list, middle_criteria = [], [], [], [], [], [], [], [], [], []
    moxco_start_epoch, once, HYPERGRAD_FLAG, state, prev_loss = [], True, None, {}, 0
    
    prev = {}
    for n, p in model.named_parameters():
        prev[n] = p.clone()

    edge_of_stability_bound = 2*(1+0.99)/((1 + 2*0.9)*(learning_rate))

    #run the training
    for e in range(1, iterations+1):
        loss = 0.
        _preds = predict(model, train_X, train_Y)
        loss_train.append(mean_squared_error(train_Y.data.numpy(), _preds.data.numpy()))

        t, prev = train_moxco_hypergradient(model, loss_fn, train_X, train_Y, prev, learning_rate, state, hyper_lr = hyper_lr, alpha=alpha, beta=beta, hyper_gradient=HYPERGRAD_FLAG)
        loss += t

        with torch.no_grad():
            grad_list = np.zeros(len([_ for _ in model.parameters()]))
            grad_ord, grad_norm = None, 0
            count = 0
            for _, p in model.named_parameters():
                tmp = p.grad.data.cpu().detach().numpy()
                grad_list[count] = np.linalg.norm(tmp)**2
                count += 1

            grad_ord = np.array(grad_list)
        grad_norm = np.sqrt(np.sum(grad_ord)) #np.mean(grad_ord)#np.linalg.norm(grad_ord)**2

        grad_norm_list.append(grad_norm)
        if is_hessian:
                largest_eigenval_estimate = eigenvalues(model, loss_fn, train_X, train_Y)
                largest_eigenval_term = largest_eigenval_estimate
                print("---- LARGEST EIGEN ----","Estimate: ", largest_eigenval_estimate)
        else:
            largest_eigenval_term = 0
        
        
        # f_x = abs(float(loss)-1) + (grad_norm**2) + (largest_eigenval_term/edge_of_stability_bound) #*10# *0.1
        # middle = 1 if (largest_eigenval_term - edge_of_stability_bound) > 0 else abs(largest_eigenval_term - edge_of_stability_bound)/edge_of_stability_bound
        middle = abs(largest_eigenval_estimate - 1.5*edge_of_stability_bound)/(1.5*edge_of_stability_bound)
        f_x = abs(float(loss)-0.64) + (grad_norm**2) + middle 
        p_window = np.exp(-f_x* tau)
        print("*"*30, f"window prob: {p_window}  f_x: {f_x}  grad_norm sq: {(grad_norm)**2}   Epoch: {e}  Eignevalue: {largest_eigenval_term/edge_of_stability_bound}", "*"*30)
        
        goodness_scores.append(p_window)
        eigenvalues_list.append(middle) #largest_eigenval_term/edge_of_stability_bound
        suboptimal_loss_list.append(abs(float(loss)-0.64))
        if p_window > eta:
            print("---------------------------------------WARM START-----------------------------------------")
            alpha = 0.01
            beta = 0.01
           
            moxco_start_epoch.append(e)
            HYPERGRAD_FLAG=True
            if once == True:
                print("----------------- HYPER GRAD -------------------")
                state = collections.defaultdict(dict)
                for n, p in model.named_parameters():
                    state[n]['ut_grad'] = torch.tensor(0.)
                    state[n]['alpha_t'] = hyper_lr
                    state[n]['grad'] = p.grad
            once = False

        
        preds = predict(model, train_X, train_Y)
        _preds_val = predict(model, test_X, test_Y)
        with torch.no_grad():
            # loss_train.append(mean_squared_error(train_Y.data.numpy(), preds.data.numpy()))
            loss_val.append(mean_squared_error(test_Y.data.numpy(), _preds_val.data.numpy()))
            mse_true.append(mean_squared_error(Y_true, preds.data.numpy()))
            mse_true_test.append(mean_squared_error(Y_true_test, _preds_val.data.numpy()))

        plot_loss.append(abs(float(loss)-1))
        print("Epoch %02d, loss = %f" % (e, loss))

        moxco_start_ = e if moxco_start_epoch == [] else moxco_start_epoch[0]

        # converging criteria
        print((abs(float(loss) - prev_loss)), grad_norm)
        if grad_norm**2 < 1e-4 and abs(prev_loss - loss) <= 1e-8: # always this
            print("converging criteria met .... ")
            break
        prev_loss = float(loss)

    print("moxco started: ", moxco_start_)
    
    # if moxco_start_epoch != []:
    #     plot_small_vals(loss_train[moxco_start_epoch[0]-1:], moxco_start_, iterations)
        
    return plot_loss, preds, _preds_val, loss_train, loss_val, mse_true, mse_true_test, moxco_start_, seed, grad_norm_list, eigenvalues_list, suboptimal_loss_list, goodness_scores


