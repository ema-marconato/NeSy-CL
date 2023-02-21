import itertools
import copy
from exp_best_args import *

def launch_addition_mnist(args):
    # define setting
    args.dataset="pair-seq-mnist"
    args.n_epochs=25
    args.c_sup=0
    args.buffer_size=1000
    args.batch_size=64
    args.minibatch_size=64

    # set project
    args.project="MNIST-ADD-SEQ"
    args.val_search = False
    args.csv_log = True

    # other settings
    hyperparameters=[
            ['joint', 'naive', 'er', 'derpp', 'cool'],  #strategy
            ['cbm', 'nesy'], # model
            [42],
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.version, args1.seed = element
        if args1.version == 'cbm':     args1=set_best_args_sequential_cbm(args1)
        elif args1.version == 'nesy':  args1=set_best_args_sequential_nesy(args1)
        else:                          NotImplementedError('Wrong choice')
        print(args1, '\n')
        args_list.append(args1)
    return args_list

def launch_shortcut_mnist(args):
    args.dataset="shortcut-mnist"
    args.n_epochs=100
    args.buffer_size=1000
    args.batch_size=256
    args.minibatch_size=256
    args.c_weight=1
    args.version = 'nesy'

    args.project="MNIST-ADD-SHORTCUT"
    args.val_search = False
    args.csv_log = True

    hyperparameters=[
            ['naive', 'restart', 'derpp', 'cool'],  #strategy
            [0, 0.01, 0.1],# supervision
            [42], # seed
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        args1= copy.copy(args)
        args1.model, args1.c_sup, args1.seed = element
        args1=set_best_args_shortcut(args1)
        args_list.append(args1)
        print(args1)
        print()
    return args_list

def launch_cle4evr(args):
    args.dataset="cle4vr"
    args.n_epochs=50
    args.buffer_size=250
    args.batch_size=64
    args.minibatch_size=64
    args.c_weight= 2
    args.version = 'nesy'
    args.project="CLE4EVR"
    args.val_search = False
    args.csv_log = True
    hyperparameters=[
            ['naive', 'restart', 'er', 'derpp', 'cool'],  #strategy
            [0, 0.01, 0.1], #c_sup
            [42] #seed
        ]

    args_list=[]
    for element in itertools.product(*hyperparameters):
        print(element)
        args1= copy.copy(args)
        args1.model, args1.c_sup, args1.seed = element
        args1=set_best_args_clevr(args1)
        args_list.append(args1)
    return args_list
