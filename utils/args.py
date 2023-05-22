from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    # dataset
    parser.add_argument('--dataset',default="pair-seq-mnist", type=str, choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    # model settings
    parser.add_argument('--version', type=str, default="nesy",help='Version of the model',
                        choices=['nesy', 'cbnm', 'baseline',"set_transformer"])
    parser.add_argument('--model', type=str,default="cool", help='Model name.', choices=get_all_models())
    parser.add_argument('--c_sup', type=float, default=0, help='Fraction of concept supervision on concepts')
    parser.add_argument('--l_weight', type=float, default=1, help='Weight associated to label loss')
    parser.add_argument('--c_weight', type=float, default=1, help='Weight associated to concept loss')
    # optimization params
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--optim_wd', type=float, default=0., help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0., help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0, help='optimizer nesterov momentum.')
    # learning hyperams
    parser.add_argument('--n_epochs',default=1, type=int, help='Number of epochs per task.')
    parser.add_argument('--batch_size', type=int,default=64, help='Batch size.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None, help='The random seed.')
    parser.add_argument('--notes', type=str, default=None, help='Notes for this run.')
    parser.add_argument('--non_verbose', action='store_true')
    parser.add_argument('--csv_log', default=True, action='store_true', help='Enable csv logging')
    parser.add_argument('--tensorboard',default=True, action='store_true', help='Enable tensorboard logging')
    parser.add_argument('--project', type=str, default="NeSy-CL", help='wandb project')
    parser.add_argument('--wandb', type=str, default=None,  help='Enable wandb logging -- set name of project')
    parser.add_argument('--val_search', action='store_true', help='Used to evaluate on the validation set for hyperparameters search')
    parser.add_argument('--posthoc', action='store_true', help='Used to evaluate on the validation set for hyperparameters search')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, default=250, help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,default=64, help='The batch size of the memory buffer.')
