def set_best_args_sequential_nesy(args1):
    # default strategies
    if args1.model == 'restart':
        args1.lr = 10**-3
    if args1.model == 'naive':
        args1.lr = 10**-4
    if args1.model == 'joint':
        args1.lr = 10**-3
    # reguralization strategies
    if args1.model == 'si':
        args1.c = 1
        args1.xi = 0.9
    if args1.model == 'ewc_on':
        args1.e_lambda = 0.7
        args1.gamma = 1.0
    if args1.model == 'lwf':
        args1.alpha = 1
        args1.softmax_temp = 2.0
    # replay methods
    if args1.model == 'er':
        args1.lr = 5*10**-4
    if args1.model == 'der':
        args1.lr = 10**-3
        args1.alpha = 0.5
    if args1.model == 'derpp':
        args1.lr = 10**-3
        args1.alpha = 0.5
        args1.beta = 1
    # concept-replay methods
    if args1.model == 'dcr':
        args1.lr =10**-3
        args1.alpha = 0.5
    if args1.model == 'cool':
        args1.lr = 5*10 **-4
        args1.alpha = 5
    return args1

def set_best_args_sequential_cbm(args1):
    if args1.model == 'restart':
        args1.lr = 5 * 10**-4
    if args1.model == 'naive':
        args1.lr = 10 ** -4
    # reguralization strategies
    if args1.model == 'si':
        args1.c = 1
        args1.xi = 0.9
    if args1.model == 'ewc_on':
        args1.e_lambda = 0.7
        args1.gamma = 1.0
    if args1.model == 'lwf':
        args1.alpha = 1
        args1.softmax_temp = 2.0
    # replay methods
    if args1.model == 'er':
        args1.lr = 10 ** -4
    if args1.model == 'der':
        args1.lr = 10**-3
        args1.alpha = 0.5
    if args1.model == 'derpp':
        args1.lr = 10**-3
        args1.alpha = 0.5
        args1.beta = 1
    # concept-replay methods
    if args1.model == 'dcr':
        args1.lr = 5*10**-4
        args1.alpha = 2
    if args1.model == 'cool':
        args1.lr = 10 **-3
        args1.alpha = 2
    return args1

def set_best_args_shortcut(args1):
    if args1.model == 'restart':
        args1.lr = 5*10**-4
    if args1.model == 'naive':
        args1.lr = 10**-4
    # reguralization strategies
    if args1.model == 'ewc_on':
        args1.e_lambda = 0.7
        args1.gamma = 1.0
    if args1.model == 'lwfshort':
        args1.alpha = 1
        args1.softmax_temp = 2.0
    # replay methods
    if args1.c_sup == 0:
        if args1.model == 'derpp':
            args1.lr = 10**-2
            args1.alpha = 0.5
            args1.beta = 1
        # concept-replay methods
        if args1.model == 'dcr':
            args1.lr = 0.005
            args1.alpha = 10
        if args1.model == 'cool':
            args1.lr = 0.005
            args1.alpha = 10

    elif args1.c_sup == 0.01:
        if args1.model == 'derpp':
            args1.lr = 0.005
            args1.alpha = 0.5
            args1.beta = 1
        # concept-replay methods
        if args1.model == 'dcr':
            args1.lr = 0.005
            args1.alpha = 0.5
        if args1.model == 'cool':
            args1.lr = 10 ** -3
            args1.alpha = 0.5

    elif args1.c_sup == 0.1:
        if args1.model == 'derpp':
            args1.lr = 0.005
            args1.alpha = 0.5
            args1.beta = 1
        # concept-replay methods
        if args1.model == 'dcr':
            args1.lr = 0.005
            args1.alpha = 0.5
        if args1.model == 'cool':
            args1.lr = 10 ** -3
            args1.alpha = 0.5
    return args1



def set_best_args_clevr(args1):
    if args1.model == 'restart':
        args1.lr = 5*10**-4
    if args1.model == 'naive':
        args1.lr = 10**-4
    if args1.model == 'der':
        args1.alpha = 0.5
    if args1.model == 'derpp':
        args1.beta = 1
    if args1.model == 'si':
        args1.c = 1
        args1.xi = 0.9
    if args1.model == 'ewc_on':
        args1.e_lambda = 0.7
        args1.gamma = 1.0
    if args1.model == 'lwf':
        args1.alpha = 1
        args1.softmax_temp = 2.0
    return args1