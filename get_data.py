# code for generating non-parameteric regression data to fit a 2 layer NN

# data has many spurious local minima
"""
f(x) = a1x+b1 x>0.5 || a2x+b2 x < 0.5
regression: y = f(x) + noise (0,1 noise)
"""

import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt

seed = 40
np.random.seed(seed)
def base_function_f(x, a_1=2,b_1=3, a_2=8, b_2=0):
    """ f(x) function with f_0(0) = 0"""
    if x >= 0.5:
        f = a_1*x + b_1
    else:
        f = a_2*x + b_2
    return f

def function_f(x):
    # https://stackoverflow.com/questions/71566171/how-to-plot-a-triangular-wave-using-a-piecewise-function-in-python
    """ f(x) function with f_0(0) = 0"""
    dec = (x % 1) - 0.5
    f = (np.abs(dec) - 0.25) * 3
    return f
    # return 1
    
def function_f_linear(x):
    return 2*x + 1

def nonuniform_piecewise(x):
    dec = (x % 1) - 0.5
    if x <= -1.5:
        f = (np.abs(dec) - 0.25) * 12
    elif -0.5 >= x > -1.5:
        f = (np.abs(dec) - 0.25) * 6
    elif 0 >=x > -0.5:
        f = (np.abs(dec) - 0.25) * 3
    else:
        f = (np.abs(dec) - 0.25) * 1.5
    return f

def doppler(x):
    """
    Parameters
    ----------
    x : array-like
        Domain of x is in (0,1]
 
    """
    if not np.all((x >= 0) & (x <= 1)):
        raise ValueError("Domain of doppler is x in (0,1]")
    return 4*np.sqrt(x*(1-x))*np.sin((2.1*np.pi)/(x+.05))

def heterogenous_smoothness(x, mode="func1"):
    if mode == "heterogeneous":
        if x < -1 :
            f = -2*x + 1
        elif x == -1:
            f = -x - 1
        elif 0 >= x > -1:
            f = 2*x + 0.5
        else:
            f = 2*np.sin(2/(0.2+x)) 
        return f

    elif mode == "func1":
        if -2 <= x < -0.5:
            f = 1+2.55- 4.5*x
        elif -0.5 <= x < 1: 
            f = 6-0.75+ 4.5*x
        elif 1<= x:
            f = 1+ 5.5 - 2*x
        return f

    elif mode == "func2":
        f = 2*np.sin(2/(1+x)) 
        return f

    elif mode == "kaiqi":
        f = np.sin(4/(x+0.01)) + 1.5
        return f
    
    elif mode == "func3":
        if -2 <= x < -0.5:
            f = 2.55- 2*x
        elif -0.5 <= x < 0.5: 
            f = 2-0.75+ 2*x
        elif 0.5 <= x < 1: 
            f = 5.5 - 2*x
        elif 1<= x:
            f = 5.5 - 2*x
        return f

def vary_dopler_v2_fcn(x):
    from kc_data import spline_wave
    # if x < 0.2:
    #     x1 = x
    #     n = 0
    # elif x < 0.4:
    #     x1 = x - 0.2
    #     n = 1
    # else:
    #     x1 = (x - 0.4) / 3
    #     n = 2
    # if x1 < 0.02:
    #     x2 = x1 / 0.02
    # elif x1 < 0.06:
    #     x2 = (x1-0.02) / 0.04
    # elif x1 < 0.12:
    #     x2 = (x1-0.06) / 0.06
    # else:
    x1 = (x - 0.4) / 3
    n = 2
    x2 = (x1-0.12) / 0.08
    return 5*spline_wave(x2, n)

def vary_dopler_v3_fcn(x):
    from kc_data import spline_wave
    if x < 0.2:
        x1 = x
        n = 1
    else:
        x1 = (x - 0.2) / 4
        n = 3
    if x1 < 0.02:
        x2 = x1 / 0.02
    elif x1 < 0.06:
        x2 = (x1-0.02) / 0.04
    elif x1 < 0.12:
        x2 = (x1-0.06) / 0.06
    else:
        x2 = (x1-0.12) / 0.08
    return 2.5*spline_wave(x2, n)

def simple(x):
    f = np.where(x < 0.5, x * 2, x * -2 + 2)
    return f

def regression_function(input_x, func = None, noise="gaussian", std=1):
    """ regression function y = f(x) + noise, input in R, 1-dimensional """
    f_x = [func(x) for x in input_x]

    center_function = lambda x: x - x.mean()
    input_x = np.array(center_function(input_x))

    if noise == "gaussian":
        mu, sigma = 0, std
        noise = np.random.normal(mu, sigma, input_x.shape)
    else:
        noise = 0

    def simple_noise_model():
        f = np.random.normal(0,1)
        if f > 0.5:
            return 1*[1]*len(input_x)
        else:
            return -1*[1]*len(input_x)

    y = f_x + noise
    y_true = f_x
    return np.array(y), y_true

def function_controlflow(func_name="piecewise_simple"):
    if func_name == "piecewise_simple":
        f = simple
    if func_name == "doppler":
        f = vary_dopler_v3_fcn
    if func_name == "heterogenous_smoothness":
        f = heterogenous_smoothness
    if func_name == "linear":
        f = function_f_linear
    return f

def generate_data(samples=1000, func="heterogenous_smoothness"): #piecewise_simple "heterogenous_smoothness"
    """ 
        samples: size of x vector
        return x: input vector of size = #samples, dimension = 1
    """
    np.random.seed(123)#123 40 123 80 #17896489#33
    x = np.random.random_sample(samples)# 4.5 for func 2/hetero, 2 for func 2 # 3 for func 1 # 5.5#3 (latest) #*6 | 4
    
    # for func 1
    # center_function = lambda x: x - x.mean()
    # x = np.array(center_function(x))
    
    # y, y_ground_truth = regression_function(x, , std=0.8) #heterogenous_smoothness
    f = function_controlflow(func_name=func)
    y, y_ground_truth = regression_function(x, f, std=0.8)
    
    # # for normal points.
    center_function = lambda x: x - x.mean()
    x = np.array(center_function(x))
    
    return x, y, y_ground_truth


def gen_doppler_points(n_sample=100):
    def dopler_fcn(x):
        return np.sin(4/(x + 0.01)) + 1.5

    # x = np.linspace(0, 1, n_sample)
    np.random.seed(98)# 123 80 #17896489#33
    x = np.random.random_sample(n_sample)
    center_function = lambda x: x - x.mean()
    x = np.array(center_function(x))

    y = np.array([dopler_fcn(xi) for xi in x])
    yt = y + np.random.normal(0, 0.5, size=y.shape)
    return x, y, list(yt)


def visualize_data(x,y, y_true):
    plt.scatter(x,y, color="orange")
    xs, ys = zip(*sorted(zip(x, y_true)))
    plt.plot(xs,ys)
    plt.grid()
    plt.savefig("input_data.png", bbox_inches='tight', dpi=120)
    plt.close()

# x, y_observed, y_true = generate_data()
# visualize_data(x, y_observed, y_true)
