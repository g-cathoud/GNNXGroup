from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pickle
import torch
import math
import json
import os

from torch import nn

from datasets.collate import collate_fn

from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

from matplotlib import cm
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, Draw
from rdkit import Chem


def load_data(file_path):
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    
    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)


def test_model(args, model, prepare_input_function, test_loader, train_median=0, train_mad=1):

    model.eval()
    total_loss = 0.0

    # Lists to store target and predicted values
    all_targets = []
    all_predictions = []
    contributions_model = []

    loss_l1 = nn.L1Loss()

    for batch_idx, data in enumerate(test_loader):
        # Transfer data to the appropriate device
        y = data[args['target']].to(args['device'], args['dtype'])
        y = (y - train_median) / train_mad
        all_targets.append(y.cpu().detach())

        # Making the input according to the model
        input = prepare_input_function(data, args)

        # Forward pass: compute the model output
        if args['model_type'].startswith('original'):
            pred, _ = model(input)
        else:
            pred, contributions_list = model(input)
            contributions_model.extend([[contribution.tolist() for contribution in contributions] for contributions in contributions_list])
        all_predictions.append(pred.cpu().detach())

        # Compute the loss using MAE
        loss = loss_l1(pred, y)

        # Accumulate the loss
        total_loss += loss.item()

    # Calculate average loss over the batches
    avg_loss = total_loss / len(test_loader)

    # Concatenate all batches for targets and predictions
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    if args['model_type'].startswith('original'):
        return avg_loss, all_targets, all_predictions, None
    else:
        return avg_loss, all_targets, all_predictions, contributions_model   
        
        
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


def get_gnn_data(test_loader, model_file, args):

    model, prepare_inputs_function, args = get_model_and_args(args)

    checkpoint = torch.load(f'models/Z_trained_models/{model_file}', map_location=args['device'])

    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    train_median = checkpoint['scaling_params']['median_train']
    train_mad    = checkpoint['scaling_params']['mad_train']

    avg_loss, model_true, model_pred, contributions = test_model(
        args, model, prepare_inputs_function, test_loader, train_median, train_mad)
    
    print(f'MAE from test function: {avg_loss}')

    return model_true, model_pred, contributions

def calculate_metrics(y_true, y_pred):

    # Number of samples
    n = y_true.size(0)

    # Mean of true values
    y_mean = torch.mean(y_true)

    # Calculate RMSE
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)

    # Calculate MAE
    mae = torch.mean(torch.abs(y_true - y_pred))

    # Calculate R²
    ss_total = torch.sum((y_true - y_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return {
        'RMSE': rmse.item(),
        'MAE': mae.item(),
        'R²': r2.item()
    }

def normalize_contributions(contributions):

    # Calculate the maximum absolute value, ensuring it's at least 1 to avoid division by zero
    max_abs_value = max(abs(num) for num in contributions)

    # Normalize the sublist
    normalized_sublist = [num / max_abs_value for num in contributions]

    return normalized_sublist

def generate_image(mol, mol_weights, filename):

    # Atoms
    mycm = cm.PiYG

    AllChem.Compute2DCoords(mol)

    fig = Draw.MolToMPL(mol, coordScale=1.5, size=(250, 250))
    x, y, z = Draw.calcAtomGaussians(mol, 0.05, step=0.01, weights=mol_weights)
    maxscale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))

    # this does the coloring
    img = fig.axes[0].imshow(z, cmap=mycm, interpolation='bilinear', origin='lower', extent=(
        0, 1, 0, 1), vmin=-maxscale, vmax=maxscale, alpha=0.75)

    # this draws 10 contour lines
    levels = np.linspace(-maxscale, maxscale, 11)
    levels = levels[levels != 0]  # Remove zero
    fig.axes[0].contour(x, y, z, levels=levels, colors='k', alpha=0.5)
    fig.axes[0].axis('off')

    # Save the figure as a PDF file
    plt.savefig(filename + '.svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":

    parser = ArgumentParser(description="Arguments to run the experiment.")
    parser.add_argument("--image_dir",    default='images/contribution_plots', type=str)
    parser.add_argument("--job_id",       type=str)
    parser.add_argument("--split_type",   default="scaffold")
    parser.add_argument("--batch_size",   default=8, type=int)

    args, unknown = parser.parse_known_args()
    args = vars(args)

    with open('./training_data.json', 'r') as f:
        models_information = json.load(f)

    for idx, model_file in enumerate(models_information['reaction_models']):


        print(
            f'Processing model: {model_file[:-4]} ({idx + 1} out of {len(models_information["reaction_models"])})')

        dataset, model_name, model_type, _, _ = model_file.split('_')

        with open(f'./datasets/{dataset}/data/benson_groups.pkl', 'rb') as f:
            benson_groups = pickle.load(f)

        with open(f'./datasets/{dataset}/data/atomic_groups.pkl', 'rb') as f:
            atomic_groups = pickle.load(f)

        with open(f"./datasets/{dataset}/data/reactions.pkl", 'rb') as f:
            reactions = pickle.load(f)

        with open(f"./datasets/{dataset}/data/test_indices_random.pkl", 'rb') as f:
            test_indices = pickle.load(f)

        datadir = 'datasets/{dataset}/data/'

        test_reactions = [reactions[i] for i in test_indices]
        test_data = collate_fn(test_reactions)

        test_dataset  = Subset(reactions, test_indices)
        test_loader   = DataLoader(test_dataset,  batch_size=32, shuffle=False,  num_workers=0, collate_fn=collate_fn)

        test_molecules = []
        for i in test_indices:
            test_molecules.extend(reactions[i]['reactant'])
            test_molecules.extend(reactions[i]['product'])

        max_charge = max([molecule['max_charge'] for molecule in test_molecules if molecule is not None])
        num_species = len(set(charge for molecule in test_molecules if molecule is not None for charge in molecule['charges']))

        images_dir = './images/contribution_plots'

        args =  {'model_name': model_name,
            'model_type': model_type,
            'dataset': dataset,
            'target': 'reaction_enthalpy',
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'dtype': torch.float32,
            'max_charge': max_charge,
            'num_species': num_species}
        
        model_true, model_pred, contributions = get_gnn_data(test_loader, model_file, args)

        metrics = calculate_metrics(model_true, model_pred)

        print(f"Model: {model_file}")
        print(f"RMSE: {metrics['RMSE']}")
        print(f"MAE: {metrics['MAE']}")
        print(f"R²: {metrics['R²']}")
        print()

        folder = f'{dataset}_{model_name}_{model_type}'
        os.mkdir(os.path.join(images_dir, folder))
        
        for react_idx in range(len(test_indices)):
            mol_idx = 0

            for molecule in test_reactions[react_idx]['reactant']:
                w = normalize_contributions(contributions[react_idx][mol_idx][:molecule['num_atoms']])
                image_tag = f'{react_idx}_{mol_idx}_reactant.svg'
                generate_image(molecule['mol'], w, os.path.join(images_dir, folder, image_tag))
                mol_idx += 1
                
            for molecule in test_reactions[react_idx]['product']:
                w = normalize_contributions(contributions[react_idx][1][:molecule['num_atoms']])
                image_tag = f'{react_idx}_{mol_idx}_product.svg'
                generate_image(molecule['mol'], w, os.path.join(images_dir, folder, image_tag))
                mol_idx += 1






