import math
import numpy as np

# cifar-10 tensor is of size 16x16x3x3 => 3 or 4 size depending on implementation.
prev_pred = 0
class Expert():
    def __init__(self, mean=0):
        self.prediction = mean
        self.loss = 0
        self.count = 0
        self.weight = 0

def init_experts(pool, n):
    count = 0
    for k in range(int(math.floor(math.log2(n))) + 1):
        stop = ((n + 1) >> k) - 1
        # print("k: ", k, "stop: ", stop)
        if stop < 1:
            break
        elist = [Expert() for i in range(1, stop + 1)] # creates list of geometric covers with active experts.
        # pool[k] = (elist) #pool.append(elist)
        # print("list of intervals", len(elist))
        pool.append(elist)
        count += stop
    return count, pool # had to add pool to return in order to get it updated in the main file

def get_awake_set(idx, t, n):
    """
    for 1d version
    """
    for k in range(math.floor(math.log2(t)) + 1):
        # print("Inside awake set function: ", k)
        i = (t >> k)
        if ((i + 1) << k) - 1 > n:
            idx.append(-1)
        else:
            idx.append(i)

def get_forecast(awake_set, pool, normalizer, pool_size):
    """
    online version
    """
    output = 0
    normalizer = 0
    for k in range(len(awake_set)):
        if awake_set[k] == -1:
            continue
        i = awake_set[k] - 1
        if pool[k][i].weight == 0:
            pool[k][i].weight = 1.0 / pool_size
            pool[k][i].prediction = prev_pred
            # print("pool pred og ", k, i, "updated to", prev_pred)
        output = output + (pool[k][i].weight * pool[k][i].prediction)
        
        normalizer = normalizer + pool[k][i].weight
        # print("NORMALIZER UPDATE:", normalizer, output)
    return output / normalizer, normalizer

def compute_losses(awake_set, pool, losses, y, B, n, sigma, delta):
    norm = 2 * (B + sigma * math.sqrt(math.log(2 * n / delta))) * (B + sigma * math.sqrt(math.log(2 * n / delta)))
    for k in range(len(awake_set)):
        if awake_set[k] == -1:
            losses.append(-1)
        else:
            i = awake_set[k] - 1
            loss = (y - pool[k][i].prediction) *  (y - pool[k][i].prediction) / norm
            losses.append(loss)
    # return losses

def update_weights_and_predictions(awake_set, pool, losses, normalizer, y):
    norm = 0
    # compute new normalizer
    for k in range(len(awake_set)):
        if awake_set[k] == -1:
            continue
        i = awake_set[k] - 1
        norm = norm + pool[k][i].weight * math.exp(-losses[k])
        # print("i", i, pool[k][i].weight, pool[k][i].loss)
    # update weights and predictions
    for k in range(len(awake_set)):
        if awake_set[k] == -1:
            continue
        i = awake_set[k] - 1
        # print(k, "------------", pool[k][i].weight, math.exp(-losses[k])* normalizer /norm)
        pool[k][i].weight = pool[k][i].weight * math.exp(-losses[k]) *(normalizer / norm)
        pool[k][i].prediction = ((pool[k][i].prediction * pool[k][i].count) + y) / (pool[k][i].count + 1)
        pool[k][i].count = pool[k][i].count + 1

def run_aligator_original(n, y, index, sigma, B, delta):
    ''' Directly fom Xunadng's code'''
    global prev_pred
    prev_pred = 0
    estimates = [0] * n
    pool = [[] for _ in range(n)]
    pool_size = init_experts(pool, n)
    print(pool_size)
    for t in range(n):
        print("t: ", t)
        normalizer = 0
        awake_set = []
        idx = index[t]
        y_curr = y[idx]
    
        j = get_awake_set(awake_set, idx + 1, n)
        print(j, awake_set)

        output = get_forecast(awake_set, pool, normalizer, pool_size)
        estimates[idx] = output

        losses = []
        losses = compute_losses(awake_set, pool, losses, y_curr, B, n, sigma, delta)
        print(losses)

        update_weights_and_predictions(awake_set, pool, losses, normalizer, y_curr)
        prev_pred = y_curr
    return estimates

def run_aligator(n, t, y, pool, pool_size, sigma, B, delta):
    ''''
    This is online version of aligator. This estimates the gradients as they come and uses prev update information alongside.
    There are 4 main globals pool, pool_size, prev_pred, losses. They are being tracked from previous iterations.
    '''
    global prev_pred
    print("prev_pred inside:", prev_pred)
    # global pool etc for running this file for debugging
    estimates = 0 # initalized every func call
    normalizer = 0 # initalized every func call
    awake_set = [] # initalized every func call
    y_curr = y

    # print("-----", awake_set, t + 1, n)
    get_awake_set(awake_set, t + 1, n)
    # print("Awake set:", awake_set)

    output, normalizer = get_forecast(awake_set, pool, normalizer, pool_size) # normalizer keep getting updated
    estimates = output
    losses = [] # initalized every func call
    compute_losses(awake_set, pool, losses, y_curr, B, n, sigma, delta) # globaly update losses that are being tracked from prev state.
    # print("Awake set:", awake_set, "losses", losses)

    update_weights_and_predictions(awake_set, pool, losses, normalizer, y_curr)
    prev_pred = y_curr # prev update
    
    return estimates

def run_aligator_offline(n, y, sigma, B, delta):
    ''' This is offline version of aligator for optimization. It is called after every epoch where it forgets all the past state info'''
    global prev_pred
    prev_pred = 0
    estimates = np.zeros(n)
    pool = []
    pool_size, pool = init_experts(pool, n)
    print(pool_size, pool)
    index = np.arange(0, n)
    
    for t in range(n):
        normalizer = 0
        awake_set = []
        idx = index[t]
        y_curr = y[idx]

        # print("-----", awake_set, t + 1, n)
        get_awake_set(awake_set, idx + 1, n)
        print("Awake set:", awake_set)

        output, normalizer = get_forecast(awake_set, pool, normalizer, pool_size)
        estimates[idx] = output

        losses = []
        losses = compute_losses(awake_set, pool, losses, y_curr, B, n, sigma, delta)
        print("losses", losses)

        update_weights_and_predictions(awake_set, pool, losses, normalizer, y_curr)
        prev_pred = y_curr
    return estimates

def aligator_algorithm_offline(n, y, sigma, B, delta):
    ''' THIS IS A OFFLINE VERSION OF ALIGATOR FOR OUR CASE
        time horizon: n
        loss function params: B,sigma,delta
        index: expert indices?
        y: entity that is to be denoised
    '''
    res = run_aligator_offline(n, y, sigma, B, delta)
    return res

def aligator_algorithm(n, t, y, pool, pool_size, sigma, B, delta):
    '''
        time horizon: n
        loss function params: B,sigma,delta
        index: expert indices?
        y: entity that is to be denoised
    '''
    res = run_aligator(n, t, y, pool, pool_size, sigma, B, delta)
    return res
#-----------------------------------------------#


# Case 1: similar to Xunadong
# time_range, sigma, B, delta = 15 , 0.9, 0.25, pow(10,-4) # len is num layers.
# y = np.random.normal(0,sigma,15)
# print(y, np.arange(0,time_range))
# denoised_grad_norm_squared = run_aligator(time_range, y, np.arange(0,time_range), sigma, B, delta)
# print(denoised_grad_norm_squared)
# run aligator2d_4: best results - test_alig.py

# Case 2: Online optimization, # t = 7
# n = 15
# time_range, sigma, B, delta = n, 0.9, 0.50, pow(10,-2)
# pool = []
# pool_size = init_experts(pool, n)
# print("--", pool_size)
# prev_pred = 0
# losses = []
# for t in range(15):
#     y = 10*(t+1)
#     print("#-----------------------------------------------#")
#     denoised_grad_norm_squared = aligator_algorithm(time_range, t, y, sigma, B, delta)
#     print(denoised_grad_norm_squared, prev_pred)

# Case 3: Offline optimization, to be run on all accmulated mini-batch gradientnorm square
# y = [100, 11, 0.4, 0.3, 0.9]
# n = 5
# time_range, sigma, B, delta = n, 0.25, 4.5, pow(10,-4)
# denoised_grad_norm_squared = run_aligator_edited(n, y, sigma, B, delta)
# print(denoised_grad_norm_squared)