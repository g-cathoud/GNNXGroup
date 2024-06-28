import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pickle
import torch
import json
import os

from datasets.collate import collate_fn
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from torch import nn


def load_data(file_path):

    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)

    elif file_path.endswith('.pkl'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    elif file_path.endswith('.pth'):
        return torch.load(file_path)

    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)


def test_gnn_model(args, model, prepare_input_function, test_loader, train_median=0, train_mad=1):

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
        if args['model_type'] != 'original':
            pred, atom_contributions = model(input)
            contributions_model.extend(atom_contributions.tolist())
        else:
            pred, _ = model(input)
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

    if args['model_type'] == 'original':
        return avg_loss, all_targets, all_predictions, None
    else:
        return avg_loss, all_targets, all_predictions, contributions_model


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

    return rmse.item(), mae.item(), r2.item()


def get_gnn_data(test_loader, args):

    if args['model_name'] == 'egnn':

        from models.egnn.egnn import prepare_inputs_egnn, get_egnn_model
        prepare_inputs_function = prepare_inputs_egnn
        model, args = get_egnn_model(args)

    elif args['model_name'] == 'schnet':

        from models.schnet.schnet import prepare_inputs_schnet, get_schnet_model
        prepare_inputs_function = prepare_inputs_schnet
        model, args = get_schnet_model(args)

    else:
        raise ValueError

    checkpoint = load_data(f'models/Z_trained_models/{args["model_file"]}')

    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    train_median = checkpoint['scaling_params']['median_train']
    train_mad = checkpoint['scaling_params']['mad_train']

    avg_loss, model_true, model_pred, contributions_model = test_gnn_model(
        args, model, prepare_inputs_function, test_loader, train_median, train_mad)
    rmse, mae, r2 = calculate_metrics(model_true, model_pred)

    model_file_name, _ = os.path.splitext(args['model_file'])

    results = {
        'model': model_file_name,
        'true': model_true,
        'pred': model_pred,
        'RMSE': rmse,
        'MAE': mae,
        'MAE_testfunction': avg_loss,
        'R2': r2
    }

    with open(f'{args["results_dir"]}/{model_file_name}_results.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    if contributions_model is not None:
        with open(f'{args["results_dir"]}/{model_file_name}_contributions.pkl', 'wb') as f:
            pickle.dump(contributions_model, f,
                        protocol=pickle.HIGHEST_PROTOCOL)


def get_regression_data(args):
    """ 
    Load the necessary data accoring to the different splits and all the traning function.
    """

    df = load_data(args["df_file"])
    groups = load_data(args["groups_file"])

    train_indices = load_data(args["train_indices_file"])
    val_indices = load_data(args["val_indices_file"])
    test_indices = load_data(args["test_indices_file"])

    train_df = df.iloc[train_indices]
    val_df   = df.iloc[val_indices]
    train_df = pd.concat([train_df, val_df])
    test_df  = df.iloc[test_indices]

    property_columns = ['mu', 'alpha', 'homo', 'lumo',
                        'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']

    groups_names = [group["smarts"] for group in groups]
    group_counts = {}
    for group in groups_names:
        group_counts[group] = train_df[train_df[group] > 0].shape[0]
    group_items = list(group_counts.items())
    present_groups = [key for key, value in group_items if value != 0]

    columns_to_check = [col for col in test_df.columns if col not in present_groups
                        and col not in property_columns and col != 'smiles']
    for col in columns_to_check:
        test_df = test_df[test_df[col] == 0]
    x_test_df = test_df[present_groups]
    y_test_df = test_df[property_columns]

    bundle = load_data(f"models/Z_trained_models/{args['model_file']}")

    model = bundle['model']
    model_params = bundle['scaling_params']

    regression_pred = model.predict(x_test_df)
    regression_true = (y_test_df[args['target']].values - model_params['median_train']) / model_params['mad_train']

    model_true, model_pred = torch.tensor(
        regression_true), torch.tensor(regression_pred)

    rmse, mae, r2 = calculate_metrics(model_true, model_pred)

    model_file_name, _ = os.path.splitext(args['model_file'])

    results = {
        'model': model_file_name,
        'true': model_true,
        'pred': model_pred,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    with open(os.path.join(args["results_dir"], f'{model_file_name}_results.pkl'), 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_pred_true(true, pred, file_name):

    plt.figure(figsize=(3.26, 3.1))  # Adjusted for a better visualization size

    plt.scatter(true, pred, alpha=0.5, s=10,
                label='Regression model', color='blue')
    plt.xlabel('True Value', fontsize=10)  # Adding x-label for clarity
    plt.ylabel('Predicted Value', fontsize=10)

    # Add a legend
    plt.legend(loc='upper left', fontsize=8)

    # Set tick intervals based on the range of values
    max_range = max(max(true), max(pred), max(true), max(pred))
    min_range = min(min(true), min(pred), min(true), min(pred))

    plt.plot([min_range, max_range], [min_range, max_range], linestyle='--', color='red', lw=1, label='X=Y')
    plt.xlim(min_range, max_range)
    plt.ylim(min_range, max_range)

    plt.xticks([])
    plt.yticks([])

    # Set the aspect of the plot to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(f'./results/true_pred_{file_name}.eps', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":

    training_data = load_data('./training_data.json')

    for idx, model in enumerate(training_data['gnn_models']):
        print(
            f'Processing model: {model} ({idx + 1} out of {len(training_data["gnn_models"])})')

        if model.endswith('custom_train'):
            dataset, model_name, model_type, target, date, time, _, _ = model.split(
                '_')
        else:
            dataset, model_name, model_type, target, date, time, = model.split(
                '_')

        with open(f"./datasets/{dataset}/data/molecules_scaffold_split.pkl", 'rb') as f:
            molecules = pickle.load(f)

        with open(f"./datasets/{dataset}/data/test_indices_scaffold.pkl", 'rb') as f:
            test_indices = pickle.load(f)

        test_dataset = Subset(molecules, test_indices)
        test_loader = DataLoader(test_dataset,  batch_size=32,
                                 shuffle=False,  num_workers=0, collate_fn=collate_fn)

        max_charge = max([molecules[i]['max_charge'] for i in test_indices])
        num_species = len(
            set(charge for i in test_indices for charge in molecules[i]['charges']))

        model_args = {
            'model_file': model + '.pth',
            'model_name': model_name,
            'model_type': model_type,
            'dataset': dataset,
            'target': target,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'dtype': torch.float32,
            'max_charge': max_charge,
            'num_species': num_species,
            'results_dir': './results/'}

        get_gnn_data(test_loader, model_args)

        del molecules, test_dataset, test_loader

    for idx, model in enumerate(training_data['regression_models']):
        print(
            f'Processing model: {model} ({idx + 1} out of {len(training_data["regression_models"])})')

        model_name, dataset, group_type, target, _, split_type = model.split('_')

        datadir = f'./datasets/{dataset}/data'

        model_args = {
            'model_file': model + ".pkl",
            'target': target,
            'df_file': f'./datasets/{dataset}/data/data_{group_type}_groups.csv',
            'groups_file': f'./datasets/{dataset}/data/{group_type}_groups.pkl',

            'train_indices_file': f'{datadir}/train_indices_scaffold.pkl',
            'val_indices_file': f'{datadir}/val_indices_scaffold.pkl',
            'test_indices_file': f'{datadir}/test_indices_scaffold.pkl',

            'results_dir': './results/'}

        get_regression_data(model_args)

    results_files = []
    contributions_files = []

    # Iterate over all the files in the directory
    for file in os.listdir('./results/'):
        # Check if the file ends with the specified ending
        if file.endswith('results.pkl'):
            results_files.append(file)
        elif file.endswith('contributions.pkl'):
            contributions_files.append(file)

    models = []
    RMSE = []
    MAE = []
    R2 = []

    for file in results_files:

        model_file_name, _ = os.path.splitext(file)
        results = load_data(f'./results/{file}')
        models.append(results['model'])
        RMSE.append(results['RMSE'])
        MAE.append(results['MAE'])
        R2.append(results['R2'])
        get_pred_true(results['true'], results['pred'], file)

    # Plotting RMSE
    plt.figure(figsize=(5, 0.5*len(models)))
    bars = plt.barh(models, RMSE, color='skyblue')
    plt.xlabel('RMSE')
    plt.xscale('log')

    # Adding values to the bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')

    plt.savefig('./results/rmse.eps', bbox_inches='tight')
    plt.close()

    # Plotting MAE
    plt.figure(figsize=(5, 0.5*len(models)))
    bars = plt.barh(models, MAE, color='skyblue')
    plt.xlabel('MAE')
    plt.xscale('log')

    # Adding values to the bars
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', va='center')

    plt.savefig('./results/mae.eps', bbox_inches='tight')
    plt.close()

    # Plotting R2
    plt.figure(figsize=(5, 0.5*len(models)))
    bars = plt.barh(models, R2, color='skyblue')
    plt.xlabel('R2')
    plt.xlim(0, 1)
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.savefig('./results/r2.eps', bbox_inches='tight')
    plt.close()