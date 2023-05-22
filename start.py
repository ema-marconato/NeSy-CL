import os
import sys

conf_path = os.getcwd() + "."
sys.path.append(conf_path)

from argparse import ArgumentParser
import torch
from utils.continual_training import train
import setproctitle
import uuid
import datetime
import socket
import importlib
import socket
from utils.training import train
from utils.offline_training import train_offline
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_dataset
from utils.losses import get_loss
from models import get_model
from utils.conf import set_random_seed
from six.moves import urllib
from experiments import *
from utils.conf import get_device


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

def parse_args(args):
    parser = ArgumentParser(description='NeSy-CL', allow_abbrev=False)
    parser.add_argument('--model', type=str,default="der", help='Continual-model name.', choices=get_all_models())
    torch.set_num_threads(4)

    # load args related to seed etc.
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    get_parser = getattr(mod, 'get_parser') # LOAD THE PARSER SPECIFIC OF THE MODEL, WITH ITS SPECIFICS
    parser = get_parser()
    args = parser.parse_args() # this is the return

    if args.seed is not None:
        set_random_seed(args.seed)
    else: set_random_seed(42)

    return args

def start_main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()
    print('\n',args,'\n')

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

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
    
    if get_device == 'cuda':
        model.net= model.net.cuda()

    # set job name
    setproctitle.setproctitle('{}_{}_{}_{}'.format(args.version, args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    # perform posthoc evaluation/ cl training/ joint training
    if model.NAME != 'joint': train(model, dataset, args)
    else: train_offline(model, dataset, args)
    
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

    start_main(args)

