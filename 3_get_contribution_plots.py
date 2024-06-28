from argparse import ArgumentParser
from matplotlib import cm
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import AllChem, Draw
from rdkit import Chem

import shutil
import pickle
import torch
import math
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from datasets.collate import collate_fn

from utils.dataset_utils import progress_bar

from typing import List, Dict, Any, Tuple, Callable


def get_model_and_args(args: Dict[str, Any]) -> Tuple[torch.nn.Module, Callable, Dict[str, Any]]:

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


def get_model_contributions(model, data_loader, prepare_inputs_function, args):

    contributions_model = []

    for batch_idx, data in enumerate(data_loader):

        input = prepare_inputs_function(data, args)
        pred, atom_contributions = model(input)
        contributions_model.extend(atom_contributions.tolist())

    return contributions_model


def filter_contributions(data, contributions, model_type):

    n_atoms = data['num_atoms']
    n_groups = [len(temp) for temp in data['benson_groups']]

    contributions_filtered = []

    for idx in range(len(data['num_atoms'])):

        if model_type == 'group':
            ids_groups = data['benson_groups_ids'][idx][:n_groups[idx]]
            cont_filtered = [contributions[idx][i] for i in ids_groups]

        elif model_type == 'atomic':
            cont_filtered = contributions[idx][:n_atoms[idx]]

        else:
            raise ValueError

        contributions_filtered.append(cont_filtered)

    return contributions_filtered


def get_cos_sim_and_mae(cont1, cont2):

    def calculate_cosine_simuliarity(vector1: List, vector2: List):

        cos_sim = np.dot(vector1, vector2) / \
            (np.linalg.norm(vector1) * np.linalg.norm(vector2))

        return cos_sim

    cosine_sim = []
    mae = []

    for idx in range(len(cont1)):

        cosine_sim.append(calculate_cosine_simuliarity(cont1[idx], cont2[idx]))
        absolute_error = np.abs(np.array(cont1[idx]) - np.array(cont2[idx]))
        mae.append(np.mean(absolute_error))

    return cosine_sim, mae


def get_top_bottom_values(data_list, top_n, reverse=True):

    top_items = sorted(enumerate(data_list),
                       key=lambda x: x[1], reverse=reverse)[:top_n]
    # key=lambda x: abs(x[1]), reverse=reverse)[:top_n]
    indexes, values = zip(*top_items) if top_items else ([], [])

    return list(values), list(indexes)


def normalize_contributions(contributions, idx, n_atoms):
    # Get the sublist of numbers up to n_atoms
    sublist = contributions[idx][:n_atoms]

    # Calculate the maximum absolute value, ensuring it's at least 1 to avoid division by zero
    max_abs_value = max(max(abs(num) for num in sublist), 1)

    # Normalize the sublist
    normalized_sublist = [num / max_abs_value for num in sublist]

    return normalized_sublist


def generate_image(mol: Chem.Mol, mol_weights: List[float], filename: str, cs, mae):

    anotation_text = f'Cos sim: {cs:.2f}, MAE: {mae:.2f}'

    # Atoms
    mycm = cm.PiYG

    AllChem.Compute2DCoords(mol)

    fig = Draw.MolToMPL(mol, coordScale=1.5, size=(250, 250))
    # the values 0.02 and 0.01 can be adjusted for the size of the molecule
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

    fig.axes[0].text(0.5, -0.1, anotation_text, ha='center', va='center',
                     transform=fig.axes[0].transAxes, fontsize=32, color='black')

    # Save the figure as a PDF file
    plt.savefig(filename + '.svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def generate_distribution(data: List[float], filename: str):
    mean = np.mean(data)
    std = np.std(data)

    file = os.path.basename(filename)

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=50, color='skyblue')

    # Calculate text position
    text_x = plt.gca().get_xlim()[1] * 0.75
    text_y = plt.gca().get_ylim()[1] * 0.9

    # Add mean and std as text
    plt.text(text_x, text_y, f'\mu: {mean:.2f}',
             fontsize=12, color='red', verticalalignment='top')
    plt.text(text_x, text_y * 0.85,
             f'\sigma: {std:.2f}', fontsize=12, color='green', verticalalignment='top')

    plt.xlabel('Values')
    plt.ylabel('Frequency')

    plt.savefig(filename + '.svg', bbox_inches='tight')
    plt.close()


def get_data(args, model_group_cont, contribution_type):

    print('Loading the data...')

    # Load the data
    with open(f'./datasets/{args["dataset"]}/data/molecules_{args["split_type"]}_split.pkl', 'rb') as f:
        molecules = pickle.load(f)
    with open(f'./datasets/{args["dataset"]}/data/mols.pkl', 'rb') as f:
        mols = pickle.load(f)
    with open(f'./datasets/{args["dataset"]}/data/test_indices_{args["split_type"]}.pkl', 'rb') as f:
        test_indices = pickle.load(f)

    test_molecules = [molecules[i] for i in test_indices]
    test_mols = [mols[i] for i in test_indices]
    test_data = collate_fn(test_molecules)

    if contribution_type == 'group':
        lr_group_cont  = test_data[f"{args['target']}_benson_coefs"].tolist()
    elif contribution_type == 'atomic':
        lr_group_cont = test_data[f"{args['target']}_atomic_coefs"].tolist()

    print('Getting the metrics...')

    model_group_cont_filtered = filter_contributions(
        test_data, model_group_cont, contribution_type)
    lr_group_cont_filtered = filter_contributions(
        test_data, lr_group_cont, contribution_type)
    cosine_sim_group, mae_group = get_cos_sim_and_mae(
        model_group_cont_filtered, lr_group_cont_filtered)

    print(f'Mean cosine similarity: {np.mean(cosine_sim_group):.2f}')
    print(f'Mean MAE: {np.mean(mae_group):.2f}')

    _, idx_top_cossim_group = get_top_bottom_values(cosine_sim_group,  500)
    _, idx_top_corr_grop = get_top_bottom_values(mae_group,   100)

    idx_top_unique = list(set(idx_top_cossim_group + idx_top_corr_grop))

    print('Making the images...')

    if args['custom_train']:
        dirtag = f'{args["dataset"]}_{args["target"]}_{args["model_name"]}_{contribution_type}_custom'
    else:
        dirtag = f'{args["dataset"]}_{args["target"]}_{args["model_name"]}_{contribution_type}'

    if not os.path.exists(os.path.join(args['image_dir'], dirtag)):
        os.makedirs(os.path.join(args['image_dir'], dirtag))
    else:
        shutil.rmtree(os.path.join(args['image_dir'], dirtag))
        os.makedirs(os.path.join(args['image_dir'], dirtag))

    generate_distribution(
        cosine_sim_group, f'{args["image_dir"]}/{dirtag}/distribution_com_sim_group')
    generate_distribution(
        mae_group, f'{args["image_dir"]}/{dirtag}/distribution_mae_group')

    for i, idx in enumerate(idx_top_unique):

        progress_bar(i, len(idx_top_unique))
        n_atoms = test_data['num_atoms'][idx]

        model_group_cont_norm = normalize_contributions(
            model_group_cont, idx, n_atoms)
        lr_group_cont_norm = normalize_contributions(
            lr_group_cont, idx, n_atoms)

        generate_image(test_mols[idx], model_group_cont_norm,
                       f'{args["image_dir"]}/{dirtag}/mol_{idx}_model', cosine_sim_group[idx], mae_group[idx])
        generate_image(test_mols[idx], lr_group_cont_norm,
                       f'{args["image_dir"]}/{dirtag}/mol_{idx}_regression', cosine_sim_group[idx], mae_group[idx])

    print('Done...')


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

    for idx, contributions_file in enumerate(models_information['contribution_models']):

        print(
            f'Processing model: {contributions_file[:-4]} ({idx + 1} out of {len(models_information["contribution_models"])})')

        if contributions_file.endswith('custom_train_contributions.pkl'):
            dataset, model_name, group_type, target, _, _, _, _, _ = contributions_file.split('_')
            custom_train = True
        else:
            dataset, model_name, group_type, target, _, _, _ = contributions_file.split('_')
            custom_train = False

        args = {"image_dir": 'images/contribution_plots',
                "results_dir":'./results',
                "split_type": "scaffold",
                "batch_size": 8,
                "dataset": dataset,
                "model_name": model_name,
                "target": target,
                "custom_train": custom_train}

        with open(os.path.join(args['results_dir'], contributions_file), 'rb') as f:
            model_group_cont = pickle.load(f)

        get_data(args, model_group_cont, group_type)