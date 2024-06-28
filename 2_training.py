import pickle
import random
import torch
import wandb
import numpy as np

from torch import optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Any, Tuple, Callable

from utils.model_utils import AlphaScheduler, train_model, train_model_custom_loss, test_model, compute_median_mad
from datasets.collate import collate_fn

torch.autograd.set_detect_anomaly(True)


def get_model_and_args(args: Dict[str, Any]) -> Tuple[torch.nn.Module, Callable, Dict[str, Any]]:

    if args['model_name'] == 'egnn':

        from models.egnn.egnn import prepare_inputs_egnn, get_egnn_model
        model, args = get_egnn_model(args)
        return model, prepare_inputs_egnn, args

    elif args['model_name'] == 'segnn':

        from models.segnn.segnn import prepare_inputs_segnn, get_segnn_model
        model, args = get_segnn_model(args)
        return model, prepare_inputs_segnn, args

    elif args['model_name'] == 'painn':

        from models.painn.painn import prepare_inputs_painn, get_painn_model
        model, args = get_painn_model(args)
        return model, prepare_inputs_painn, args

    elif args['model_name'] == 'schnet':

        from models.schnet.schnet import prepare_inputs_schnet, get_schnet_model
        model, args = get_schnet_model(args)
        return model, prepare_inputs_schnet, args

    elif args['model_name'] == 'visnet':

        from models.visnet.visnet import prepare_inputs_visnet, get_visnet_model
        model, args = get_visnet_model(args)
        return model, prepare_inputs_visnet, args

    elif args['model_name'] == 'lieconv':

        from models.lieconv.lieconv import prepare_inputs_lieconv, get_lieconv_model
        model, args = get_lieconv_model(args)
        return model, prepare_inputs_lieconv, args

    else:
        raise ValueError


def experiment(
        model: torch.nn.Module,
        prepare_inputs_function: Callable,
        args: Dict[str, Any],
        train_loader: DataLoader,
        val_loader: DataLoader,
        median_train: float,
        mad_train: float
) -> None:

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"\nExperiment start:\n")
        file.write(f"Contructing the model\n")

    optimizer = optim.Adam(
        model.parameters(), args['lr'], weight_decay=args['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args['epochs'])

    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"\nModel architecture:\n {model} \n")
        file.write(f"# of parameters: {total_param} \n\n")
        file.write(f"Training the model... \n")

    if args['custom_train']:
        alpha_scheduler = AlphaScheduler(0.5, 0, args['epochs'])
        train_model_custom_loss(args, model, prepare_inputs_function, optimizer, lr_scheduler,
                                train_loader, val_loader, alpha_scheduler, median_train, mad_train)
    else:
        train_model(args, model, prepare_inputs_function, optimizer,
                    lr_scheduler, train_loader, val_loader, median_train, mad_train)

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"\nTesting the model...\n")

    mae, true, pred = test_model(
        args, model, prepare_inputs_function, test_loader, median_train, mad_train)

    data = [[x, y] for (x, y) in zip(true, pred)]
    table = wandb.Table(data=data, columns=["true", "pred"])
    wandb.log(
        {f"{args['target']}/test_data": wandb.plot.scatter(table, "true", "pred")})

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f'\nThe testing MAE was: {mae}')
        file.write(f"\nDone with {args['target']}!\n")


if __name__ == "__main__":

    parser = ArgumentParser(description="Arguments to run the experiment.")

    parser.add_argument("--dataset",         default='qm9',      type=str)
    parser.add_argument("--model_name",      default='schnet',   type=str)
    parser.add_argument("--model_type",      default='original', type=str)
    parser.add_argument("--overfit_test",    default=False,      type=bool)
    parser.add_argument("--wandb_mode",      default='online',   type=str)
    parser.add_argument("--slurm_id",        default='None',     type=str)

    # Training parameters
    parser.add_argument("--target",          default="H")
    parser.add_argument("--split_type",      default="scaffold")
    parser.add_argument("--custom_train",    default=False)
    parser.add_argument("--log_gradients",   default=False)
    parser.add_argument("--epochs",          default=1000,  type=int)
    parser.add_argument("--batch_size",      default=32,    type=int)
    parser.add_argument("--log_freq",        default=20,    type=int)
    parser.add_argument("--patience",        default=10,    type=int)
    parser.add_argument("--weight_decay",    default=1e-16, type=float)
    parser.add_argument("--seed",            default=0,     type=int)
    parser.add_argument("--sample",          default=None,  type=int)

    args, unknown = parser.parse_known_args()
    args = vars(args)

    torch.manual_seed(args['seed'])

    if args['custom_train'] and args['model_type'] != 'group':
        raise ValueError("Custom training is only available for group models.")

    # Format the current date and time
    run_tag = f"{args['dataset']}_{args['model_name']}_{args['model_type']}_{args['target']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    name_tag = f"{args['dataset']}_{args['model_name']}_{args['model_type']}_{args['target']}"

    if args['overfit_test']:
        run_tag = run_tag + '_overfit_test'
        name_tag = name_tag + '_overfit_test'

    if args['custom_train']:   
        run_tag = run_tag + '_custom_train'
        name_tag = name_tag + '_custom_train'

    args['report_file'] = f"./reports/{run_tag}.txt"
    args['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args['dtype'] = torch.float32
    args['run_tag'] = run_tag

    # Load the processed data
    with open(f"./datasets/{args['dataset']}/data/molecules_{args['split_type']}_split.pkl", 'rb') as f:
        molecules = pickle.load(f)

    # Load the training and validation indexes
    with open(f"./datasets/{args['dataset']}/data/train_indices_{args['split_type']}.pkl", 'rb') as f:
        train_indices = pickle.load(f)
    with open(f"./datasets/{args['dataset']}/data/val_indices_{args['split_type']}.pkl", 'rb') as f:
        val_indices = pickle.load(f)
    with open(f"./datasets/{args['dataset']}/data/test_indices_{args['split_type']}.pkl", 'rb') as f:
        test_indices = pickle.load(f)

    if args['sample'] and not args['overfit_test']:
        train_indices = random.sample(train_indices, args['sample'])
        val_indices = random.sample(val_indices,   args['sample']//10)
        test_indices = random.sample(test_indices,  args['sample']//10)

    if args['overfit_test']:
        train_dataset = Subset(molecules, random.sample(train_indices, 1))
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False,  num_workers=0, collate_fn=collate_fn)
        val_dataset = Subset(molecules, random.sample(train_indices, 1))
        val_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=False,  num_workers=0, collate_fn=collate_fn)
        test_dataset = Subset(molecules, random.sample(test_indices, 1000))
        test_loader = DataLoader(
            test_dataset,  batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)

    else:
        # Create the dataloaders
        train_dataset = Subset(molecules, train_indices)
        train_loader = DataLoader(
            train_dataset, batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)
        val_dataset = Subset(molecules, val_indices)
        val_loader = DataLoader(
            train_dataset, batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)
        test_dataset = Subset(molecules, test_indices)
        test_loader = DataLoader(
            test_dataset,  batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)

    prop_data_train = [molecules[i][args['target']] for i in train_indices]

    args['max_charge'] = max([molecules[i]['max_charge']
                             for i in train_indices])
    args['num_species'] = len(
        set(charge for i in train_indices for charge in molecules[i]['charges']))

    if args['target'] in ['lumo', 'homo', 'gap']:
        args['lr'] = 5e-4
    else:
        args['lr'] = 5e-4

    if args['overfit_test']:
        median_train, mad_train = 0, 1
    else:
        median_train, mad_train = compute_median_mad(prop_data_train)

    wandb.init(
        project="benson-groups-xai",
        mode=args['wandb_mode'],
        dir="./reports/",
        name=name_tag,
        tags=[args['dataset'], args['model_name'],
              args['model_type'], args['target'], 
              'official']
    )

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"\nStarting experiment with {args['target']}...\n")
        file.write(f"The experiment will be conducted on: {args['device']}\n")
        file.write(f"The slurm job id is: {args['slurm_id']}\n")


    model, prepare_inputs_function, args = get_model_and_args(args)

    wandb.config.update(args)

    experiment(model, prepare_inputs_function, args,
               train_loader, val_loader, median_train, mad_train)

    wandb.finish()
