import os
import sys
from argparse import ArgumentParser
import torch
conf_path = os.getcwd() + "."
sys.path.append(conf_path)
from utils.continual_training import train
import setproctitle
import uuid
import datetime
import socket
import importlib
import socket
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from utils.continual_training import train as ctrain
from datasets import get_dataset
from utils.losses import get_loss
from models import get_model
from utils.training import train
from utils.offline_training import train_offline
from utils.conf import set_random_seed
from utils.main import lecun_fix
from utils.posthoc_analysis import MNIST_posthoc
from experiments import *


def parse_args(args):
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str,default="der", help='Continual-model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true', help='Loads the best arguments for each method, '
                             'dataset and memory buffer.') # ok, come si usa?
    torch.set_num_threads(4)

    # load args related to seed etc.
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)


    get_parser = getattr(mod, 'get_parser') # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    parser = get_parser()
    parser.add_argument('--project', type=str, default="NeSy-CL", help='wandb project')
    args = parser.parse_args() # this is the return

    if args.seed is not None:
        set_random_seed(args.seed)

    return args





def start_main(args=None):
    lecun_fix() # CHE FA?
    if args is None:
        args = parse_args()

    print(args)

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    # Load model and loss
    backbone = dataset.get_backbone(args.version)
    loss     = get_loss(args)

    model    = get_model(args, backbone, loss, dataset.get_transform())


    # SAVE A BASE MODEL FOR ALL CL-STRATEGIES

    PATH = f'data/{args.dataset}-{args.version}-start.pt'
    if os.path.exists(PATH):
        model.net.load_state_dict(torch.load(PATH))
    else:
        print('Created',PATH, '\n')
        torch.save(model.net.state_dict(), PATH)

    model.net= model.net.cuda()

    # set job name
    setproctitle.setproctitle('{}_{}_{}_{}'.format(args.version, args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    # perform posthoc evaluation/ cl training/ joint training
    if isinstance(dataset, ContinualDataset):
        if args.posthoc: MNIST_posthoc(model, dataset, loss)
        elif model.NAME != 'joint': train(model, dataset, args)
        else: train_offline(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)

    print('\n ### Closing ###')

if __name__ == '__main__':
    #start_main()
    args = parse_args(None)
    import submitit
    executor = submitit.AutoExecutor(folder="./nesycontinual", slurm_max_num_timeout=30)
    executor.update_parameters(

            mem_gb=2,
            gpus_per_task=1,
            tasks_per_node=1,  # one task per GPU
            cpus_per_gpu=10,
            nodes=1,
            timeout_min=60,  # max is 60 * 72
            # Below are cluster dependent parameters
            slurm_partition="SLURM_PARAMETER",
            slurm_signal_delay_s=120,
            slurm_array_parallelism=4)



    experiments=launch_shortcut_mnist(args)+ launch_cle4evr(args)+launch_addition_mnist(args) #+launch_mnist_cbm_offline(args)
    # executor.update_parameters(name="cbm")
    # jobs = executor.map_array(start_main,experiments)

    start_main(experiments)

