'''Binary neural nets via prox operations'''

# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

# hidden_layer_size = 44
# experiment.log_parameter("hidden_layer_size", hidden_layer_size)

import argparse
import os
import copy
import time
import json
import logging
import collections
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import models
import random
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
from reg import *
from writer import *

from InertialProxSGD import *
from HessianFlow import hessianflow as hf
from aligator_resnet import init_experts, run_aligator, get_awake_set
# seed = 242
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


track_grad_norm = {}
global WARM_START
WARM_START = False
min_grad, max_grad = np.inf, 0
z_i_norm_square = 0
variance_for_sigma_inside_aligator = 0.1

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--save_all', action='store_true',
                    help='save model at every epoch')
parser.add_argument('--tb_dir', type=str, default=None,
                    help='TensorBoard directory')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--binary_reg', default=0.0, type=float,
                    help='Binary regularization strength')
parser.add_argument('--reg_rate', default=0.0, type=float,
                    help='Regularization rate')
parser.add_argument('--adjust_reg', action='store_true',
                    help='Adjust regularization based on learning rate decay')
parser.add_argument('--projection_mode', default=None, type=str,
                    help='Projection / rounding mode')
parser.add_argument('--freeze_epoch', default=-1, type=int,
                    help='Epoch to freeze quantization')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--binarize', action='store_true',
                    help='Load an existing model and binarize')
parser.add_argument('--binary_regime', action='store_true',
                    help='Use alternative stepsize regime (for binary training)')
parser.add_argument('--ttq_regime', action='store_true',
                    help='Use alternative stepsize regime (for ttq)')
parser.add_argument('--no_adjust', action='store_true',
                    help='Will not adjust learning rate')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
########### additional parsers #############
parser.add_argument('--alpha', default=0.9, type=float,
                    help='alpha for inertial prox')
parser.add_argument('--beta',  default=0.9, type=float,
                    help='beta for inertial prox')
parser.add_argument('--alpha_exp_config',  default=1, type=int,
                    help='alpha experiment config')
parser.add_argument('--resetting_window',  default=0.868, type=float,
                    help='resetting window')
parser.add_argument('--reset_alpha',  default=0.5, type=float,
                    help='resetting window alpha')
parser.add_argument('--reset_beta',  default=0.5, type=float,
                    help='resetting window beta')
parser.add_argument('--temperature',  default=0.025, type=float,
                    help='temperature')
############ latest added parser args #############
# parser.add_argument('--optim_switch', default="ADAM", type="str", 
#                     help="optim switch name")
parser.add_argument('--hyper_lr', default=0.001, type=float, 
                    help="Hypergradient lr")
parser.add_argument('--random_beta', default=False, type=bool, 
                    help="beta is random")



def eigenvalues(model, criterion, val_loader):
    get_data = True
    for data, target in val_loader:
        # finish the for loop otherwise there is a warning
        if get_data:
            inputs = data
            targets = target
            get_data = False

    eigenvalue, eigenvec = hf.get_eigen(model, inputs, targets, criterion, cuda = True, maxIter = 3, tol = 1e-3)
    print('\nCurrent Eigenvalue based on Test Dataset: %0.2f' % eigenvalue)
    print("Eigenvalues..", eigenvalue, len(eigenvec))
    return eigenvalue

def prox_reg(model, lambda_):
    regularizer = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if if_binary(n): 
                # diff = torch.sub(p.data, torch.sign(p.data))
                diff = torch.sub(p, torch.sign(p))
                # regularizer += torch.norm(diff, 1)#lambda_ * torch.norm(diff, 1)
                regularizer += torch.norm(diff, 1)
        regularizer = lambda_ * regularizer
        regularizer = regularizer.to(device=args.gpus[0]).type(args.type)
    return regularizer

def alpha_config(epoch):
    if args.alpha_exp_config == 1:
        args.alpha = 0.0033 * epoch
    elif args.alpha_exp_config == 2:
        args.alpha = (epoch -1) / (epoch + 1)
    elif args.alpha_exp_config == 3:
        args.alpha = (2*epoch - 1) / (2*epoch + 1)
    elif args.alpha_exp_config == 4:
        args.alpha = (epoch - 1) / (epoch + 2)
    elif args.alpha_exp_config == 5:
        args.alpha = (epoch - 2) / (epoch + 2)
    elif args.alpha_exp_config == 6:
        args.alpha = (3*epoch - 1) / (3*epoch + 1)  
    return args.alpha

def alpha_restart_scheme(epoch):
    warmstart_regime = {0: {'alpha':0.3},
         80: {'alpha':0.3},
         120: {'alpha':0.3}} #0.4 everywhere
    
    if epoch == warmstart_regime.keys():
        args.alpha = warmstart_regime[epoch]['alpha']
    return args.alpha

def eigenvalue_clipping(eigenvalue, EOS_bound, max_diff):
    """  20, 30 50 """
    diff = EOS_bound - eigenvalue
    if diff > max_diff:
        max_diff = diff
        return eigenvalue, max_diff
        
    else:
        clip_value = 1000000
        return clip_value, max_diff

def write_to_csv(alist, blist, clist):
    data = pd.DataFrame({'lr_list':alist})
    data['val_prec1_bin'] = blist
    data['eigenvals'] = clist
    data.to_csv("results_val_lr.csv", sep='\t')

def _write_to_csv(keys, vals, filename="largest_eigs.csv"):
    adict = {}
    for i, k in enumerate(keys):
        adict[k] = vals[i]
    data = pd.DataFrame(adict)
    data.to_csv(filename, sep='\t')

def hypergradient(hyper_lr, sub_state):
    # print("^^^^^^^^^^", "old grad:" , torch.norm(sub_state['ut_grad']))
    ht = sub_state['grad']*sub_state['ut_grad']
    alpha_t = sub_state['alpha_t'] - hyper_lr*ht
    ut = -alpha_t*sub_state['grad']
    ut_grad = -sub_state['grad']
    return ut, ut_grad, alpha_t

# make runs deterministic
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

def main():
    global args, best_prec1
    global WARM_START
    global pool 
    global pool_size
    global z_i_norm_square 
    global variance_for_sigma_inside_aligator

    global state
    best_prec1 = 0
    WARM_START = False
    args = parser.parse_args()
    hyper_params = {
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "alpha": args.alpha,
    "beta": args.beta,
    "projection_mode": args.projection_mode,
    "lambda": args.reg_rate,
    "binary lambda:": args.binary_reg,
    "alpha experiment config": args.alpha_exp_config
    }
    # experiment.log_parameters(hyper_params)

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    # logger = Logging()

    # new_results = CifarResultsLog(results_file % 'csv', results_file % 'html')
    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    writer = TensorboardWriter(args.tb_dir)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None
    logging.info("Alpha:  %s",args.alpha)
    logging.info("Beta:  %s",args.beta)
    logging.info("Reset Beta:  %s",args.reset_beta)
    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    model.cuda(device=args.gpus[0])
    print("GPU: ", args.gpus[0])
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            # results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            # args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            # best_prec1 = best_prec1.cuda(args.gpus[0])
            best_prec1 = best_prec1.cuda(args.gpus[0]) if isinstance(best_prec1, torch.Tensor) else best_prec1
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Adjust batchnorm layers if in stochastic binarization mode
    if args.projection_mode == 'stoch_bin':
        adjust_bn(model)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})

    # Adjust stepsize regime for specific optimizers
    if args.binary_regime:
        regime = {
            0: {'optimizer': 'Adam', 'lr': 1e-2},
            81: {'lr': 1e-3},
            122: {'lr': 1e-4},
        }
    elif args.ttq_regime:
        regime = {
            0: {'optimizer': 'SGD', 'lr': 0.1,
                'momentum': 0.9, 'weight_decay': 2e-4},
            80: {'lr': 1e-2},
            120: {'lr': 1e-3},
            300: {'lr': 1e-4}
        }
    elif args.optimizer == 'Adam':
        regime = {
            0 : {'optimizer': 'Adam', 'lr': args.lr},
        }
        
    elif args.optimizer == 'InertialProxSGD':
        print("****************************************")
        regime = {
            0 : {'optimizer': 'InertialProxSGD', 'lr': args.lr}
        }

    elif args.projection_mode != None:
        # Remove weight decay when using SGD, and reset momentum
        regime[0]['weight_decay'] = 0.0
        regime[0]['momentum'] = 0.0 #args.momentum

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    # if args.random_beta == False:
    #     state = ('MT19937',[1833452142, 1869000023, 3328781827, 2108883836, 1035884088,
    #         573173923, 2993901518, 3173185536,  291880974,  976271583,
    #     1565994042,  157346512, 3446618094, 3968089214, 1663104980,
    #         895265691, 1243957921, 2737464148, 1370225611, 2727628742,
    #     1538784149, 1429749668,  205531700,  389588740, 3422136880,
    #     2118383080, 4147751675, 3074696487, 2503476346, 3251517823,
    #     1773844817,  161425801, 1302205583, 3130908123, 1733281418,
    #     2358809842,  570916529, 2215121199, 2162812819, 1728973847,
    #     1723852966, 2896103147, 2457791390, 3597875980, 2721671709,
    #         240804754, 2355939162,  894943737, 2250097880,  346054565,
    #     2388389547,  114066534,  457021413, 2611078912, 1384189480,
    #     3228520468, 2804405499, 4111340439, 3379708306, 2579260054,
    #     1174743284, 3691959036, 3851538071, 2053264414, 3359966856,
    #     2437976900, 2953798934, 3542180893, 3225549025, 3448707462,
    #     3071014891, 3422372317, 1166057055, 4185356485, 1307284462,
    #     3238413201, 2313099592, 4045406694, 1417448389,  604896245,
    #     1885782732, 1924015439, 3524467033, 3610431805,  680694982,
    #     3999940438, 4123363739, 1482660056, 3858699482, 3475169048,
    #     3525032529, 1807614628, 2628671692, 2607657133, 3328840355,
    #         349695463, 2194813981, 1525970845, 1499151665, 2831230203,
    #     3927265487, 1370638542, 2301591028, 3383334195, 3709864585,
    #     2595241260, 1337074609,  195827682, 3993446732, 3894028745,
    #     3614312791, 2651794931, 1514151862,  518977313,   61691234,
    #     1099145280, 1195188757,  624868588, 1815818574, 2350963881,
    #     1692219576, 1706417293, 2957984597, 3866479785, 2666879820,
    #     3496322250, 3722237990, 3281847043, 3809762357, 3082401630,
    #     2525807594,  862709745,  782055725, 1488177839, 1360652479,
    #         291518275, 3830971507, 4026122704, 1712661606,  556835925,
    #     2709045694,  568170927,  258551391,  103473997, 3194715492,
    #         652555234, 2251254548, 1777047395,  138986002, 1691161563,
    #     1564855222, 3345408266, 2868474692,  553261634, 2060870450,
    #         879689568,  674592208, 3283996539,  502751601,   51451684,
    #         89401844, 4256098850,  756344495,  346872591, 2734727395,
    #     4175095795,  441546468, 3521627551, 3723719651, 2894150097,
    #     2898865627, 1778775357, 1771494556, 4255580232, 2468752695,
    #     4044276373, 1267101940, 3154475065,  376859223, 1842058528,
    #     1169797517, 1620209612, 3555385471, 3844959339,  620146715,
    #     2731037799, 1261210827, 1489931973, 1413960486, 2488014412,
    #         169798496,  960176222, 2182538986, 1738395657, 1782576291,
    #         264065882, 2192335243, 4035836504, 2486062199, 1539182784,
    #     2791414907, 4289876560,  961557619, 2169828167,  373511924,
    #     3398032521,  389950456, 1030186464, 1294156302, 2986467691,
    #     3329119773,  857436662, 3859306654, 1073501439, 3991472809,
    #     3730849827, 3243181884, 1874867006, 1196684274, 3018091141,
    #         992344007, 1174134768, 2878000628, 2789690109, 2530027156,
    #     2625733799, 3076827158, 3211790001,  534638878, 4220974500,
    #     4049389942, 2343504075, 3690866888, 1471964018,  376891287,
    #     4143765975, 3828328791, 4286410691, 2900574269,  408221297,
    #     1916516164,  294218216, 3824923377, 3584727394, 3818733795,
    #         164449218, 1598867991,  214472640, 4242296373, 2478720674,
    #     1583915143,  276440513, 3748134262,  366644451, 1296218137,
    #     3230867432,  479482925, 1721611148, 3005821161, 4127482328,
    #     1363010884, 4252218471, 3108208657, 1299938188, 2175827359,
    #     4171986688, 3827957743, 1967259216, 2727452209, 2679350471,
    #     4176943584,  966085659, 3034622364, 1563353710,   32825655,
    #     2109576093,   52332376, 3265891279,  839776474, 3359968442,
    #     2695890359, 1798920638, 2533258189, 4225084416, 3260229984,
    #     1583632570, 2542517422, 1482307709, 3716592593, 3199484794,
    #     4274299871,  261966373, 2130856764, 3729518572,  358632668,
    #         284037102, 2694769239,  414989575, 1881199114, 1906061801,
    #     2594841149, 3265860284,  167143035, 3762421107, 1243375462,
    #     3646534883, 3084752124, 2942937386,  633806345, 4145388965,
    #         519745593, 2023713304,  908648344, 1897681132, 1046064263,
    #     2611356410, 2244616223, 3262876014, 1376508629,    5822804,
    #     2493303643, 2039125472,  475309148, 4168212015, 1228863219,
    #     4251581422, 3330055927,  601421256, 4178307730, 1750166041,
    #         793313862, 3396579653, 2422238904, 4265956102, 4288255194,
    #         107037592, 2556074523, 1568972659, 1568333258, 3929653249,
    #     3067442071, 1417377199, 4287317657, 1651573495, 2897392905,
    #     3521333647, 1116531979, 2819280004, 1257837888, 2863836817,
    #         894731661,  305772253, 2638276959, 1272394864,  399696664,
    #     3446863230,  606463556, 1466827542, 1213732121,  401688701,
    #         707439356, 3961151441, 1560040796, 3334004471, 4190284826,
    #     2631214827, 3455359522, 3873922881, 2628438474,  651853078,
    #     3995014159,  919976698, 1741596512, 2685111991,  445964990,
    #         28137545, 4069565592, 3455745778, 3562582978, 2727362902,
    #     1606960121, 2585954864, 2578575410,  904317506, 1207068819,
    #     3352341087, 3554540595,  956501291, 4213879634,  298482220,
    #     1585926390, 3444782413, 2475194331, 1828627349, 1032305176,
    #         991175489, 2009846678,  599896040, 4234152534, 3028461736,
    #         893321372, 1707021637, 4064195795,   68783112, 3569170228,
    #         580275880, 2599760734, 2036746776, 3519757145, 1904037166,
    #     3213071323, 3384922958,  684936487,  345525116, 1247747333,
    #     2467790316, 2066345617, 3340663181, 2996708545,  840171349,
    #     3371421892, 4044926045, 2544080943, 4112086515, 2453002727,
    #     1559858464,  329781895,  987975072, 2623318185, 1444097697,
    #     3076253437, 3931171978, 1133809343, 1199232201, 1487184575,
    #     2316284108, 3624894974,  747259918, 3307569927, 2645757894,
    #     3619121028,  215327581,  227491033, 2877156432,   48113730,
    #     4208837300, 1848723040,  902017537,  236068317, 2508519746,
    #     2987856649, 1591755553,  417584073, 3773721919, 1809532713,
    #     1877087286, 3861368699, 1771693384, 2768627465,   41672419,
    #         44151005, 1507079790,  667641295, 3295476169, 3965739950,
    #     4199783573, 4215904081, 2498941441, 3263449823,  945508477,
    #     3218146197, 1034880211, 3341994416, 3119726457, 2122694445,
    #     1917909864, 4121858484,  109907016, 2364602760, 2330886461,
    #         58890498, 1489089040,  587035682,  388231030, 4081933405,
    #     3355170297, 1334235987,  587442345, 3485665938, 3561095584,
    #         271199081,  627260715,  295222396, 3147692568, 1804165711,
    #     2040240705,  847946446, 3878304740, 2890295911, 2810791953,
    #         530114437, 3667147788,  219617589,  628870153, 3175840319,
    #     3525452614, 1508637663,  434925943, 1372728166,  698046060,
    #         131438181,  381513762, 1172820168,  721128707,   80528957,
    #     3249669916,  341336300, 1250509336, 2623182719,  314853126,
    #     1403812507,   98634491, 3496488193, 2973947644, 1624465877,
    #     1596340745,  690903423, 1235005197, 2479132550, 2488528841,
    #     3995424525,  694340610,  302150238, 1248454004, 1413871868,
    #     4092884592, 1483722041, 2079792785, 1008783570, 3829232322,
    #     1281135480, 1351050257, 1825621255, 3967382222, 1617852228,
    #         832928208, 1975609141, 1555836058, 2082560862, 1326762843,
    #         221814594, 3116295954, 2892574710, 1655953580,  730853742,
    #     2578729093, 1675982671, 4176287679, 3861195259, 1235241928,
    #     3617395818, 2114413344, 2807039639,  682805202, 3365928123,
    #     2695609490, 3055000477,  164591162,  193403922, 1768455412,
    #     2781323392, 1689481878, 3191745089, 4220597702, 3154568025,
    #     3993171611, 4184403186,  242857407, 1402473221, 4059989323,
    #     1680625236, 1113885434,  759110981, 1064128673, 2640816875,
    #     2067036805, 2374739310, 1243911909, 3161703386, 3415816193,
    #     1740166866, 2524272899, 2053900196, 2122747672, 1087510884,
    #         794008799, 4139880637, 1112147203, 3148504810, 2785944335,
    #     1520295820, 1614627534, 2878983700, 2078902011,  169473105,
    #     2078276437,  905273608, 2135917716, 3724085295, 3771506668,
    #         892402835, 3396089670, 2679002986, 1219246502, 1168790578,
    #     3458798102, 1293944389,  181299378, 3333663091, 1948539123,
    #         177379574,  998836113, 2386242341,   24113825], 151, 0, 0.0)
        # np.random.set_state(state)
    
    # define samplers for obtaining training and validation batches
    train_data = get_dataset(args.dataset, 'train', transform['train'])

    # subset_size=0.25#0.25
    # num_subset = len(train_data)
    # indices = list(range(num_subset))
    # np.random.shuffle(indices)
    # print(indices)
    # split = int(np.floor(subset_size * num_subset))
    # subset_idx = indices[:split]
    st0 = np.random.get_state()
    print(st0)
    # batch size logic
    # train_loader_list, val_loader_list = [], []
    # batch_sizes_list = [32, 128, 512, 2048]#[32, 128, 512, 2048]#[2048, 512, 128, 32]
    # list_of_subsets = np.array_split(subset_idx, 4)
    # print(split, len(subset_idx))
    # print([len(k) for k in list_of_subsets])
    # for i, subset_num in enumerate(list_of_subsets):
    #     train_sampler = torch.utils.data.SubsetRandomSampler(subset_num)
    #     train_loader = torch.utils.data.DataLoader(
    #         train_data,
    #         batch_size=batch_sizes_list[i],
    #         num_workers=args.workers, pin_memory=True,
    #         sampler= train_sampler)
        
    #     val_loader = torch.utils.data.DataLoader(
    #         val_data,
    #         batch_size=batch_sizes_list[i], shuffle=False,
    #         num_workers=args.workers, pin_memory=True)
        
    #     train_loader_list.append(train_loader)
    #     val_loader_list.append(val_loader)

    #     print(len(train_loader), len(train_loader.sampler.indices), len(val_loader))
    
    
    ##
    # normal 0.25/subset code
    # train_sampler = torch.utils.data.SubsetRandomSampler(subset_idx)
    # train_loader = torch.utils.data.DataLoader(
    #     train_data,
    #     batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True,
    #     sampler= train_sampler)
    # print(num_subset, len(train_loader), split, len(train_loader.sampler.indices),  val_loader.dataset)

    # use this when full batch.
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    print(len(train_loader),  val_loader.dataset)
    prev = {}
    count=0
    xt_list, bdict = list(), dict()
    for n, p in model.named_parameters():
        prev[n] = p.clone()
        # track_grad_norm[n] = torch.zeros(p.size())
        count+=1
    
    optimizer = None
    # logging.info('training regime: %s', regime)
    
    bin_op = BinOp(model,if_binary=if_binary_tern if args.projection_mode in [
                       'prox_ternary', 'ttq'] else if_binary,
                   ttq=(args.projection_mode == 'ttq'))

    # Optionally freeze before training
    if args.freeze_epoch == 1:
        bin_op.quantize(mode='binary_freeze')
        args.projection_mode = None
    global warm_start_counter
    warm_start_counter = 0
    # Loop over epochs
    print("Reg rate: ", args.reg_rate)
    # ----------------- Initialization of pool of experts ----------------- #
    n = len(train_loader)
    pool = []
    pool_size, pool = init_experts(pool, n)
    print("pool len:", len(pool), "Pool size:", pool_size)
    print("count: ", count)
    Y_mean_estimate, y_mean_square_estimate = 0,0#np.zeros(count, dtype='float64'), 0
    near_expected_grad_list = np.zeros(count, dtype='float64')
    z_i_norm_square_list = [0]
    try:
        sign_change_list = []
        e_list, smoothed_e_list = [], []
        smoothed_vec, state = 0, {}
        for epoch in range(args.start_epoch, args.epochs):

            # experiment.log_metrics({"beta_":args.beta}, epoch=epoch)
            # experiment.log_metrics({"learning rate":args.lr}, epoch=epoch)
            # Adjust binary regression mode if non-lazy projection
            if args.projection_mode in ['prox', 'prox_median', 'prox_ternary']:
                # br = args.reg_rate 
                br = args.reg_rate * epoch
                # experiment.log_metrics({"homotopy":br}, epoch=epoch)
            else:
                br =  args.binary_reg
                print("binary", br)
             
            
            train_loss, train_plot_loss, train_prec1, train_prec5, prev, grad_list, denoised_grad_norm_squared_list = train(
                train_loader, model, criterion, prev, epoch, optimizer,
                br=br, bin_op=bin_op, projection_mode=args.projection_mode)

            # first_key = list(prev.keys())[0]
            # if epoch > 0 and torch.allclose(prev[first_key], temp_prev[first_key], atol = 0.00001):
            #     print("*********Prev did not update: ", epoch)
            
            # ----------------------------ALIGATOR --------------------------------#
            # this calculation will be per mini-batch just doing this outside of train loop
            # time_range, sigma, B, delta = len(train_loader), 0.5, max_grad, pow(10,-4) # len is num minibatches.
            # denoised_grad_norm_squared = aligator_algorithm(time_range, sqaured_norm_list, sigma, B, delta)
            # print("+++++", len(denoised_grad_norm_squared))
            # # debiased_aligator_estimation = 0
            # # for idx, bias in enumerate(grad_bias_list): # later change this to dict DS
            # #     debiased_aligator_estimation += denoised_grad_norm_squared[idx] - bias
            # #np.linalg.norm(denoised_grad_norm_squared)**2
            # denoised_grad_norm_squared = np.linalg.norm(denoised_grad_norm_squared)**2
            # print("-"*40, denoised_grad_norm_squared, B)
            
            # # the only calculation outside of train we do is for estimating as near as possible true gradient and as near as possible bias.
            # loss_bias_estimate()
            # near_expected_grad = np.mean(grad_list) 
            # print("near_expected_grad: ", near_expected_grad )
            # ----------------------------ALIGATOR --------------------------------#

            # calculating the aligator noise estimates
            near_expected_grad = near_expected_grad_list

            with torch.no_grad():
                loss_grad_snapshot = np.zeros(count)# 137 | {}
                count, grad_sq_tmp = 0, 0
                for n,p in model.named_parameters(): 
                    tmp = p.grad.data.cpu().detach().numpy() # dict object before

                    loss_grad_snapshot[count]= np.linalg.norm(tmp)
                    # grad_sq_tmp+= np.linalg.norm(p.grad.data.cpu().detach().numpy())**2
                    count+=1

                # near_expected_grad is outside element
                z_i = np.subtract(loss_grad_snapshot, near_expected_grad) # element wise subracting to get noise estimate

                k = np.linalg.norm(z_i)**2
                z_i_norm_square_list.append(k)
                z_i_norm_square = np.mean(z_i_norm_square_list)#grad_sq_tmp 

                Y_sigma_inside_aligator = 2*np.dot(z_i, loss_grad_snapshot) + k# z_i_norm_square # z_i_square faster then np.linalg.norm test
                print("Yi: ", 2*np.dot(z_i, loss_grad_snapshot))
                # sigma the one used in aligator
                # Y_mean_estimate = np.multiply(0.01, Y_sigma_inside_aligator) + np.multiply(0.99, Y_mean_estimate)
                Y_mean_estimate = 0.01*np.linalg.norm(Y_sigma_inside_aligator) + 0.99*Y_mean_estimate
                y_mean_square_estimate = 0.01*np.linalg.norm(Y_sigma_inside_aligator)**2 + 0.99*y_mean_square_estimate

                # var_list.append(y_mean_square_estimate - np.linalg.norm(Y_mean_estimate)**2)
                # variance_for_sigma_inside_aligator = np.mean(var_list)
                # variance_for_sigma_inside_aligator = y_mean_square_estimate - np.linalg.norm(Y_mean_estimate)**2
                variance_for_sigma_inside_aligator = y_mean_square_estimate - Y_mean_estimate**2
            # ----------------------------ALIGATOR --------------------------------#

            edge_of_stability_bound = 2*(1+0.99)/((1 + 2*0.9)*(args.lr))
            print("edge_of_stability_bound: ", edge_of_stability_bound)

            grad_norm = denoised_grad_norm_squared_list/len(train_loader)
            # main_term = float(train_plot_loss.item())*10 +  grad_norm * 0.1
            main_term = train_plot_loss +  grad_norm 
            if epoch < args.freeze_epoch:
                vec = eigenvalues(model, criterion, val_loader)
                smoothed_vec = 0.1*smoothed_vec + 0.9*vec
                f_x =  main_term + abs(vec/(1.5*edge_of_stability_bound) - 1) #*0.1#*#vec no scaling needed when normalized
                # f_x =  main_term + 0.1*f_x_eigen/(edge_of_stability_bound)
            else:
                f_x = main_term
                # experiment.log_metrics({"f_X":f_x, "train_loss":train_loss}, epoch=epoch)
            prob = np.exp(-args.temperature*f_x)
            # print("--", prob, "->->->", float(train_plot_loss.item()), "Exponential smoothed grad: ", grad_norm, "Epoch: ", epoch, "Alpha:", args.alpha, "Beta: ", args.beta, "fx: ", f_x)
            # print("--", prob, "->->->", float(train_plot_loss.item()), "Aligator grad: ", grad_norm, "Epoch: ", epoch, "Alpha:", args.alpha, "Beta: ", args.beta, "fx: ", f_x, vec/(edge_of_stability_bound))
            print("-- prob:", prob, "->->->", "loss: ", train_loss, train_plot_loss, float(train_plot_loss)*10, "Aligator grad: ", grad_norm, "Epoch: ", epoch, "Alpha:", args.alpha, "Beta: ", args.beta, "fx: ", f_x, "e:", abs(vec/(1.5*edge_of_stability_bound) - 1), vec/(edge_of_stability_bound))

            near_expected_grad_list = np.mean(grad_list)
            if prob > 0.872:#args.resetting_window:# and epoch >=130:#>= 130: 
                WARM_START=True
                print("**********************WARM START**********************")
                args.alpha = args.reset_alpha #0.7
                args.beta = args.reset_beta #0.2
                # args.alpha = 0
                # args.beta = 0
                warm_start_counter +=1
                # # switch to adam optimizer from here
                if warm_start_counter == 1:
                # if args.optim_switch == "adam" and warm_start_counter == 1:
                    state = collections.defaultdict(dict)
                    for n,p in model.named_parameters():
                        state[n]['ut_grad'] = torch.tensor(0.).type(args.type)
                        state[n]['alpha_t'] = args.hyper_lr
                        state[n]['grad'] = p.grad
            ######################### June 29 ##############################


            # evaluate on validation set
            val_loss, val_plot_loss, val_prec1, val_prec5, _, _, _ = validate(
                val_loader, model, criterion, epoch,
                br=br, bin_op=bin_op, projection_mode=args.projection_mode)
            
            # evaluate binarized model
            val_loss_bin, _, val_prec1_bin, val_prec5_bin, _, _, _ = validate(
                val_loader, model, criterion, epoch,
                br=br, bin_op=bin_op, projection_mode=args.projection_mode,
                binarize=True)
            
            # test_acc.append(val_loss.item())
            e_list.append(vec)
            smoothed_e_list.append(smoothed_vec)
            # remember best prec@1 and save checkpoint
            # Look at prec@1 for either binarized model or original model
            if args.binary_reg > 1e-10 or args.reg_rate > 1e-10:
                is_best = val_prec1_bin > best_prec1
                best_prec1 = max(val_prec1_bin, best_prec1)
            else:
                is_best = val_prec1 > best_prec1
                best_prec1 = max(val_prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'config': args.model_config,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'regime': regime
            }, is_best, path=save_path, save_all=args.save_all)
            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec@1 {train_prec1:.3f} \t'
                         'Training Prec@5 {train_prec5:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec@1 {val_prec1:.3f} \t'
                         'Validation Prec@5 {val_prec5:.3f} \n'
                         .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1,
                                 train_prec5=train_prec5, val_prec5=val_prec5))
            
            # experiment.log_metrics({"train_prec1":train_prec1, "val_prec1":val_prec1}, epoch=epoch)
            # experiment.log_metrics({"train_prec1":train_prec1, "val_prec1_bin":val_prec1_bin}, epoch=epoch)

            # experiment.log_metrics({"train_prec5":train_prec5, "val_prec5":val_prec5}, epoch=epoch)
            # experiment.log_metrics({"train_prec5":train_prec5, "val_prec5_bin":val_prec5_bin}, epoch=epoch)

            # experiment.log_metrics({"train_loss":train_loss, "val_loss":val_loss}, epoch=epoch)
            # experiment.log_metrics({"train_loss":train_loss, "val_loss_bin":val_loss_bin}, epoch=epoch)

            # experiment.log_metrics({"train_plot_loss":train_plot_loss, "val_plot_loss":val_plot_loss}, epoch=epoch)

            # experiment.log_metrics({"train_error1":100 - train_prec1, "val_error1":100 - val_prec1}, epoch=epoch)
            # experiment.log_metrics({"train_error1":100 - train_prec1, "val_error1_bin":100 - val_prec1_bin}, epoch=epoch)


            # experiment.log_metrics({"train_error5":100 - train_prec5, "val_error5":100 - val_prec5}, epoch=epoch)
            # experiment.log_metrics({"train_error5":100 - train_prec5, "val_error5_bin":100 - val_prec5_bin}, epoch=epoch)

            # experiment.log_metrics({"alpha":args.alpha}, epoch=epoch)
            results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, 
                        train_plot_loss=train_plot_loss, val_plot_loss=val_plot_loss,
                        val_loss_bin = val_loss_bin,
                        train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                        train_error5=100 - train_prec5, val_error5=100 - val_prec5, largest_eigen=vec)
            
            results.save()
            # custom logger for plotting bulk runs' plots
            # logger.add(random_beta=args.random_beta, epoch=epoch + 1, val_prec1_bin=val_prec1_bin, LEV=vec, lr=args.lr, zone_optim=args.optim_switch, zone_optim_lr=args.second_optim_lr)
            # logger.save()
            result_dict = {'train_loss': train_loss, 'val_loss': val_loss,
                           'train_error1': 100 - train_prec1, 'val_error1': 100 - val_prec1,
                           'train_error5': 100 - train_prec5, 'val_error5': 100 - val_prec5,
                           'val_loss_bin': val_loss_bin,
                           'val_error1_bin': 100 - val_prec1_bin,
                           'val_error5_bin': 100 - val_prec5_bin,
                           'largest_eigen': vec}
            writer.write(result_dict, epoch+1)
            writer.write(binary_levels(model), epoch+1)
            
            # Compute general quantization error
            mode = 'ternary' if args.projection_mode == 'prox_ternary' else 'deterministic'
            writer.write(bin_op.quantize_error(mode=mode), epoch+1)
            # if epoch == 298:
            #     print(bin_op.quantize_error(mode=mode))
            if bin_op.ttq:
                writer.write(bin_op.ternary_vals, epoch+1)
            

            # Optionally freeze the binarization at a given epoch
            if args.freeze_epoch > 0 and epoch+1 == args.freeze_epoch:

                if args.projection_mode in ['prox', 'lazy']:
                    bin_op.quantize(mode='binary_freeze')
                elif args.projection_mode == 'prox_ternary':
                    bin_op.quantize(mode='ternary_freeze')
                args.projection_mode = None
        results.plot("loss plot", y=['train_loss', 'val_loss'], x='epoch')
        results.plot("loss plot reg", y=['train_plot_loss', 'val_plot_loss'], x='epoch')
        results.plot("error@1", y=['train_error1', 'val_error1'], x='epoch')
        results.plot("error@5", y=['train_error5', 'val_error5'], x='epoch')
        results.plot("loss bin plot", y=['train_loss', 'val_loss_bin'], x='epoch')
        # results.plot("quantize_error", y=['quantize_error', 'val_loss'], x='epoch')
        # write_to_csv(lr_test, test_acc, e_list)
        _write_to_csv(["LEV", "smoothed_LEV"], [e_list, smoothed_e_list])
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Wrap up
    writer.close()
    return sign_change_list


def forward(data_loader, model, criterion, prev, epoch=0, training=True, optimizer=None,
            br=0.0, bin_op=None, projection_mode=None, binarize=False):
    # global WARM_START
    global min_grad, max_grad
    global track_grad_norm
    global pool
    global pool_size
    global state
    # global z_i_norm_square
    global variance_for_sigma_inside_aligator
    random_beta = args.reset_beta

    print("INSIDE", len(pool), pool_size)

    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_plot = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    # Binarize or prox-opearate the model if in eval mode
    if not(training):
        bin_op.save_params()
        if binarize:
            # bin_op.binarize()
            if projection_mode == 'prox_median':
                bin_op.quantize('median')
            elif projection_mode == 'prox_ternary':
                bin_op.quantize('ternary')
            elif projection_mode in ['prox', 'lazy']:
                bin_op.quantize('deterministic')
            elif projection_mode == 'ttq':
                bin_op.quantize('ttq')
        elif projection_mode == 'lazy':
            bin_op.prox_operator(br, 'binary')

    
    xt_dict_main, xt_list,new_prev, grad_list, var_list, denoised_grad_norm_squared_list  = {}, [], [], [],[],0
    denoised_grad_norm_squared = 0
    for i, (inputs, target) in enumerate(data_loader):
        ########## June 29 #############
        if WARM_START == False:
            args.alpha = alpha_config(i)
        
        # experiment.log_parameter("alpha", args.alpha)
        data_time.update(time.time() - end)

        if args.gpus is not None:
            target = target.cuda()

        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        
        # Binarize if projection mode is {lazy, stochastic bin} and in training
        if training:
            if projection_mode == 'lazy':
                bin_op.save_params()
                bin_op.prox_operator(br, 'binary')
            elif projection_mode == 'ttq':
                bin_op.save_params()
                bin_op.quantize('ttq')
            elif projection_mode == 'stoch_bin':
                bin_op.save_params()
                bin_op.binarize(mode='stochastic')

        if training:
            # inertial updates
            # adict, xt_dict = inertial_updates(model, prev)
            if args.random_beta and WARM_START == False:
                # np.random.set_state(state)
                random_beta = float(np.random.uniform(0.1,1,1))
            else:
                random_beta = args.beta
                # logger.save("beta_per_iteration":random_beta, "epoch":epoch, "iteration":i) 
            args.beta = random_beta
            # experiment.log_metrics({"beta_":args.beta}, epoch=epoch)
            adict, xt_dict = {}, {}
            for n,p in model.named_parameters():
                prev_iter_wts = prev[n].data.clone()
               
                x_t = p.data.clone() + (args.alpha * (p.data.clone() - prev_iter_wts))
                xt_dict[n] = x_t.clone()

                y_t =  p.data.clone() + (args.beta * (p.data.clone() - prev_iter_wts)) #random_beta
                adict[n] = p.clone()
                p.data.copy_(y_t)
            prev = adict
            xt_dict_main = xt_dict

        # forward pass
        # subgradient method
        output = model(input_var)
        loss =  criterion(output, target_var)
        # print("@@@@@@@@@@@@@", prox_reg(model, br))
        loss_plot = criterion(output, target_var).clone() * (1/args.batch_size) + prox_reg(model, br)
            
        
        # # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # losses.update(loss.data[0], inputs.size(0)) --> original
        losses.update(loss.item(), inputs.size()[0])

        # losses_plot.update(loss_plot.data[0], inputs.size(0))  --> original
        losses_plot.update(loss_plot.item(), inputs.size()[0])

        # top1.update(prec1[0], inputs.size(0)) --> original
        top1.update(prec1.item(), inputs.size()[0])
        
        # top5.update(prec5[0], inputs.size(0)) --> original
        top5.update(prec5.item(), inputs.size()[0])

        if type(output) is list:
            output = output[0]
        
        if training: 
            # compute gradient and do SGD step
            # optimizer.zero_grad()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.zero_()
                    
            
            loss.backward()
            if [x.grad for x in model.parameters()] == None:
                print("GRAD NONEEEE......")
            # copy parameters according to quantization modes
            if projection_mode in ['lazy', 'stoch_bin']:
                bin_op.restore()
                # optimizer.step()
                bin_op.clip()
            elif projection_mode == 'ttq':
                bin_op.restore()
                # optimizer.step()
                step_ternary_vals(bin_op, optimizer)
            elif projection_mode in ['prox', 'prox_median', 'prox_ternary']:
                # optimizer.step()                
                # curr_lr = optimizer.param_groups[0]['lr']
                # step(args.lr, model, xt_dict_main)
                
                # print("-----+++++++++++++-------")
                if warm_start_counter == 0: # made this change on Nov 6
                    for n, p in model.named_parameters():
                        x_t = xt_dict_main[n]
                        if p.requires_grad and p.grad is not None:
                            param_update = x_t.data.clone() - (args.lr * p.grad)
                            p.data.copy_(param_update)
                else:
                    for n, p in model.named_parameters():
                        x_t = xt_dict_main[n]
                        state[n]['grad'] = p.grad.clone()
                        ut, ut_grad, alpha_t = hypergradient(args.hyper_lr, state[n])
                        # print("!!!!!!!!!", i, torch.norm(ut), torch.norm(ut_grad), torch.norm(alpha_t), "curr grad: ", torch.norm(p.grad))
                        state[n]['alpha_t'] = alpha_t
                        state[n]['ut_grad'] = ut_grad.clone()
                        if p.requires_grad and p.grad is not None:
                            param_update = x_t.data.clone() + ut
                            p.data.copy_(param_update)
                    # experiment.log_metrics({"alpha_t":torch.norm(alpha_t).item()}, epoch=epoch)

                curr_lr = args.lr
                if projection_mode == 'prox':
                    bin_op.prox_operator(curr_lr * br, 'binary')
                elif projection_mode == 'prox_median':
                    bin_op.prox_operator(curr_lr * br, 'median')
                elif projection_mode == 'prox_ternary':
                    bin_op.prox_operator(curr_lr * br, 'ternary')
                bin_op.clip()
            else:
                # optimizer.step()
                # step(args.lr, model, xt_dict_main)
                if warm_start_counter == 0: # made this change on Nov 6
                    for n, p in model.named_parameters():
                        x_t = xt_dict_main[n]
                        if p.requires_grad and p.grad is not None:
                            param_update = x_t.data.clone() - (args.lr * p.grad)
                            p.data.copy_(param_update)
                else:
                    for n, p in model.named_parameters():
                        x_t = xt_dict_main[n]
                        state[n]['grad'] = p.grad.clone()
                        ut, ut_grad, alpha_t = hypergradient(args.hyper_lr, state[n])
                        state[n]['alpha_t'] = alpha_t
                        state[n]['ut_grad'] = ut_grad.clone()
                        if p.requires_grad and p.grad is not None:
                            param_update = x_t.data.clone() + ut
                            p.data.copy_(param_update)

        
            ########################## Aug 29 ##############################
            # exponential smoothening with square grad // verify
            count = 0
            with torch.no_grad():
                grad_norm = 0
                for n, p in model.named_parameters():
                    count+=1
                    # tmp = p.grad.data.cpu().detach().numpy()
                    # track_grad_norm[n]  = 0.1*(tmp) + 0.9*(track_grad_norm[n]) #0.01, 0.99
                    grad_norm += np.linalg.norm(p.grad.detach().cpu())**2 # instead of .cpu --> .data before
                min_grad, max_grad = min(min_grad, grad_norm), max(max_grad, grad_norm)
                    
            ########################## Sept 8 ##########################
            #------------------------- ALIGATOR -------------------------#
            # var(z^T\nabla L + z_t l2 norm sq.) calculation
            # if epoch == 0:
            #     get_awake_set()

            if WARM_START == False:   
                with torch.no_grad():
                    grad_i = np.zeros(count)
                    count = 0
                    for n,p in model.named_parameters():
                        tmp = p.grad.data.cpu().detach().numpy()
                        grad_i[count]= np.linalg.norm(tmp)
                        grad_list.append(grad_i[count])
                        count+=1 
                    grad_i_norm_square = np.linalg.norm(grad_i)**2
                    
                # zz_i_norm_square: average over past samples.
                aligator_input = grad_i_norm_square - z_i_norm_square#sigma_sq_aligator_input
                # print("aligator input check: ", grad_i_norm_square, z_i_norm_square)
                
                time_range, B, delta = len(data_loader), max_grad, pow(10,-2)
                denoised_grad_norm_squared = run_aligator(time_range, i, aligator_input, pool, pool_size, variance_for_sigma_inside_aligator, B, delta)
                # experiment.log_metrics({"Aligator denoised grad norm square":denoised_grad_norm_squared}, epoch=epoch)
                print("-"*40, denoised_grad_norm_squared, "grad_norm: ", grad_i_norm_square, "Bias z_i:", z_i_norm_square, "Aligator i/p: ", aligator_input, variance_for_sigma_inside_aligator, max_grad)

                denoised_grad_norm_squared_list+=denoised_grad_norm_squared
                denoised_grad_norm_squared_list+=grad_norm
                
            #------------------------- ALIGATOR -------------------------#

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    if not(training):
        bin_op.restore()
    return losses.avg, losses_plot.avg, top1.avg, top5.avg, prev, grad_list, denoised_grad_norm_squared_list


def train(data_loader, model, criterion, prev, epoch, optimizer,
          br=0.0, bin_op=None, projection_mode=None):
    # switch to train mode
    model.train()
    # model.eval()
    return forward(data_loader, model, criterion, prev, epoch,
                   training=True, optimizer=optimizer,
                   br=br, bin_op=bin_op, projection_mode=projection_mode)


def validate(data_loader, model, criterion, epoch,
             br=0.0, bin_op=None, projection_mode=None,
             binarize=False):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, [], epoch,
                   training=False, optimizer=None,
                   br=br, bin_op=bin_op, projection_mode=projection_mode,
                   binarize=binarize)


if __name__ == '__main__':
    main()