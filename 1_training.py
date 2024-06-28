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

from utils.model_utils import train_model, test_model, compute_median_mad
from datasets.collate import collate_fn

torch.autograd.set_detect_anomaly(True)


def get_model_and_args(args):

    if args['model_name'] == 'egnn':

        from models.egnn.egnn import prepare_inputs_egnn, get_egnn_model
        model, args = get_egnn_model(args)
        return model, prepare_inputs_egnn, args

    elif args['model_name'] == 'schnet':

        from models.schnet.schnet import prepare_inputs_schnet, get_schnet_model
        model, args = get_schnet_model(args)
        return model, prepare_inputs_schnet, args

    else:
        raise ValueError


def experiment(model, prepare_inputs_function, args, train_loader, val_loader, median_train, mad_train):

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

    train_model(args, model, prepare_inputs_function, optimizer,
                lr_scheduler, train_loader, val_loader, median_train, mad_train)

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"\nTesting the model...\n")

    mae, true, pred = test_model(
        args, model, prepare_inputs_function, test_loader, median_train, mad_train)

    data = [[x, y] for (x, y) in zip(true, pred)]
    table = wandb.Table(data=data, columns=["true", "pred"])
    wandb.log(
        {f"reaction_enthalpy/test_data": wandb.plot.scatter(table, "true", "pred")})

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f'\nThe testing MAE was: {mae}')
        file.write(f"\nDone with reaction enthalpy!\n")


if __name__ == "__main__":

    parser = ArgumentParser(description="Arguments to run the experiment.")

    parser.add_argument("--dataset",       default='rad6re')
    parser.add_argument("--model_name",    default='schnet')
    parser.add_argument("--model_type",    default='original_var1')
    parser.add_argument("--overfit_test",  default=False)
    parser.add_argument("--wandb_mode",    default='online')

    # Training parameters
    parser.add_argument("--log_gradients", default=False)
    parser.add_argument("--epochs",        default=1500,  type=int)
    parser.add_argument("--batch_size",    default=12,    type=int)
    parser.add_argument("--log_freq",      default=25,    type=int)
    parser.add_argument("--patience",      default=10,    type=int)
    parser.add_argument("--weight_decay",  default=1e-16, type=float)
    parser.add_argument("--seed",          default=0,     type=int)
    parser.add_argument("--sample",        default=None,  type=int)

    args, unknown = parser.parse_known_args()

    for arg in unknown:
        if arg.startswith(("-", "--")):
            # Dynamically add the unknown argument
            parser.add_argument(arg)

    args = vars(args)

    torch.manual_seed(args['seed'])

    # Format the current date and time
    run_tag = f"{args['dataset']}_{args['model_name']}_{args['model_type']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    name_tag = f"{args['dataset']}_{args['model_name']}_{args['model_type']}"

    if args['overfit_test']:
        run_tag = run_tag + '_overfit_test'
        name_tag = name_tag + '_overfit_test'

    args['report_file'] = f"./reports/{run_tag}.txt"
    args['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    args['dtype'] = torch.float32
    args['run_tag'] = run_tag

    # Load the processed data
    with open(f"./datasets/{args['dataset']}/data/reactions.pkl", 'rb') as f:
        reactions = pickle.load(f)

    # Load the training and validation indexes
    with open(f"./datasets/{args['dataset']}/data/train_indices_random.pkl", 'rb') as f:
        train_indices = pickle.load(f)
    with open(f"./datasets/{args['dataset']}/data/val_indices_random.pkl", 'rb') as f:
        val_indices = pickle.load(f)
    with open(f"./datasets/{args['dataset']}/data/test_indices_random.pkl", 'rb') as f:
        test_indices = pickle.load(f)

    if args['overfit_test']:
        train_dataset = Subset(reactions, random.sample(train_indices, 1))
        train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False,  num_workers=0, collate_fn=collate_fn)
        val_dataset   = Subset(reactions, random.sample(train_indices, 1))
        val_loader    = DataLoader(train_dataset, batch_size=1, shuffle=False,  num_workers=0, collate_fn=collate_fn)
        test_dataset  = Subset(reactions, random.sample(test_indices, min(200, len(test_indices))))
        test_loader   = DataLoader(test_dataset,  batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)

    else:
        # Create the dataloaders
        train_dataset = Subset(reactions, train_indices)
        train_loader  = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)
        val_dataset   = Subset(reactions, val_indices)
        val_loader    = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)
        test_dataset  = Subset(reactions, test_indices)
        test_loader   = DataLoader(test_dataset,  batch_size=args['batch_size'], shuffle=True,  num_workers=0, collate_fn=collate_fn)

    prop_data_train  = [reactions[i]['dHrxn'] for i in train_indices]

    train_molecules = []
    for i in train_indices:
        train_molecules.extend(reactions[i]['reactant'])
        train_molecules.extend(reactions[i]['product'])

    args['max_charge'] = max([molecule['max_charge'] for molecule in train_molecules if molecule is not None])
    args['num_species'] = len(set(charge for molecule in train_molecules if molecule is not None for charge in molecule['charges']))
    args['lr'] = 5e-4

    if args['overfit_test']:
        median_train, mad_train = 0, 1
    else:
        median_train, mad_train = compute_median_mad(prop_data_train)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="benson-groups-xai",
        mode=args['wandb_mode'],
        dir="./reports/",
        name=name_tag,
        tags=[args['model_name'], args['model_type'], 'reaction_enthalpy']
    )

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"\nStarting experiment...\n")
        file.write(f"The experiment will be conducted on: {args['device']}\n")

    model, prepare_inputs_function, args = get_model_and_args(args)

    experiment(model, prepare_inputs_function, args,
               train_loader, val_loader, median_train, mad_train)

    wandb.config.update(args)
    wandb.finish()
