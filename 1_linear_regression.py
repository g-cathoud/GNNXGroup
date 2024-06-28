from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_absolute_error
from utils.model_utils import load_data
from typing import List, Dict, Any, Tuple
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt


def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    # Calculate the median of the data
    median = data.median()
    # Calculate the median absolute deviation (consistently using median)
    mad = (data - median).abs().mean()
    # Scale the data
    scaled_data = (data - median) / mad
    return scaled_data, median, mad


def train_linear_model(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train_df: pd.DataFrame,
        y_test_df: pd.DataFrame,
        present_groups: List[str],
        args: Dict[str, Any]
) -> Dict[str, Dict[str, float]]:
    """
    Train a linear model on the data and return the coefficients for the groups
    """

    coefficients_dict = {}
    models = {
        'ridge': Ridge,
        'lasso': Lasso,
        'en': ElasticNet,
        'ols': LinearRegression}

    fig, axes = plt.subplots(4, 3, figsize=(6, 8))
    axes = axes.ravel()
    if args['model_type'] not in models:
        raise ValueError(f"Unsupported model_type: {args['model_type']}.")

    for i, property in enumerate(property_columns):

        print(f'Looking into property {property}', flush=True)
        model_class = models[args['model_type']]
        model = model_class(fit_intercept=False)

        y_train, median_train, mad_train  = scale_data(y_train_df[property])
        y_test = y_test_df[property]

        scaling_params = {
            'median_train': median_train,
            'mad_train': mad_train,
        }

        if args['model_type'] in ['ridge', 'lasso', 'en']:  # Alpha hyperparameter tuning
            print(f'0. Tuning hyperparameter alpha.')
            grid = {'alpha': np.arange(0.1, 1, 0.1)}
            cv = RepeatedKFold(n_splits=5, n_repeats=2,
                               random_state=args['seed'])
            search = GridSearchCV(
                model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
            results = search.fit(x_train, y_train)
            best_alpha = results.best_params_['alpha']
            best_model = model_class(alpha=best_alpha, fit_intercept=False)
        else:
            best_model = model_class(fit_intercept=False)

        print('1. Training the final model')
        best_model.fit(x_train, y_train)

        model_bundle = {
            'model': best_model,
            'scaling_params': scaling_params
        }

        # Save the model bundle using pickle
        model_path = f"models/Z_trained_models/{args['model_type']}_{args['dataset']}_{args['group_type']}_{property}_split_{args['split_type']}.pkl"

        with open(model_path, 'wb') as file:
            pickle.dump(model_bundle, file)
        
        print('2. Calculating the testing error\n-')
        y_pred = best_model.predict(x_test)
        # Scale back the values of the predictions
        y_pred = (y_pred * mad_train) + median_train
        mae = mean_absolute_error(y_test, y_pred)

        axes[i].scatter(y_test, y_pred, alpha=0.3)
        axes[i].set_title(f'{property}\nTest MAE:{mae:.2e}', fontsize=9)
        axes[i].set_ylabel('Predicted Value', fontsize=8)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        max_range = max(max(y_test), max(y_pred))
        min_range = min(min(y_test), min(y_pred))

        axes[i].set_xlim(min_range, max_range)
        axes[i].set_ylim(min_range, max_range)
        axes[i].set_aspect('equal', adjustable='box')

        coefficients_dict[property] = dict(
            zip(present_groups, best_model.coef_))
    
    plt.tight_layout()
    plt.savefig(
        f"images/{args['model_type']}_{args['dataset']}_{args['group_type']}_groups_split_{args['split_type']}.pdf", dpi=300)
    with open(f"{datadir}/coefficients_{args['model_type']}_{args['dataset']}_{args['group_type']}_groups_split_{args['split_type']}.pkl", 'wb') as f:
        pickle.dump(coefficients_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    return coefficients_dict


def get_data_for_linear_regression(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        groups: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    This function takes the train and test dataframes and the groups dictionary and returns 
    the dataframes containing the counts of the groups (only the present groups in the 
    training set are considered).
    """
    print(f"\n The total number of groups is {len(groups)} \n")
    groups_names = [group["smarts"] for group in groups]
    group_counts = {}
    for group in groups_names:
        group_counts[group] = train_df[train_df[group] > 0].shape[0]
    group_items = list(group_counts.items())
    present_groups = [key for key, value in group_items if value != 0]

    x_train_df = train_df[present_groups]
    y_train_df = train_df[property_columns]
    # The columns that are not present in the train data but are
    # present in the test data, should be removed. Otherwise,
    # this can lead to errors
    columns_to_check = [col for col in test_df.columns if col not in present_groups
                        and col not in property_columns and col != 'smiles']
    for col in columns_to_check:
        # test_df is updated to only include rows where the value in the
        # current col is equal to 0. Therefore, molecules with groups outside
        # the 'present_groups' are removed from the test set.
        test_df = test_df[test_df[col] == 0]
    x_test_df = test_df[present_groups]
    y_test_df = test_df[property_columns]

    return x_train_df, x_test_df, y_train_df, y_test_df, present_groups


def process_and_train(
        args: Dict[str, Any],
        datadir: str,
        indices_paths: Dict[str, str]
) -> Dict[str, Dict[str, float]]:
    """ 
    Load the necessary data accoring to the different splits and all the traning function.
    """
    df = load_data(f'{datadir}/data_{args["group_type"]}_groups.csv')
    groups = load_data(f'{datadir}/{args["group_type"]}_groups.pkl')

    train_indices = load_data(indices_paths['train'])
    val_indices = load_data(indices_paths['val'])
    test_indices = load_data(indices_paths['test'])

    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    train_df = pd.concat([train_df, val_df])
    test_df = df.iloc[test_indices]
    # For linear regression, the traning and validation sets are combined because we will do
    # a cross-validation and there is no need to monitor the learning: this is a regression problem

    x_train, x_test, y_train_df, y_test_df, present_groups = get_data_for_linear_regression(
        train_df, test_df, groups)
    coefficients = train_linear_model(
        x_train, x_test, y_train_df, y_test_df, present_groups, args)

    return coefficients


if __name__ == "__main__":

    parser = ArgumentParser(description="Arguments to run the experiment.")
    parser.add_argument("--seed",           default=0, type=int)
    parser.add_argument("--dataset",        default='qm9')
    parser.add_argument("--model_type",     default='ridge')
    parser.add_argument("--datadir_root",   default='./datasets/')

    args, unknown = parser.parse_known_args()
    args = vars(args)

    datadir = args['datadir_root'] + args['dataset'] + '/data/'
    indices_paths = {
        'random':   {'train': f'{datadir}/train_indices_random.pkl',
                     'val':   f'{datadir}/val_indices_random.pkl',
                     'test':  f'{datadir}/test_indices_random.pkl'},
        'scaffold': {'train': f'{datadir}/train_indices_scaffold.pkl',
                     'val':   f'{datadir}/val_indices_scaffold.pkl',
                     'test':  f'{datadir}/test_indices_scaffold.pkl'}}

    if args['dataset'] == 'qm9':
        from utils.qm9_utils import property_columns
    elif args['dataset'] == 'qmugs':
        from utils.qmugs_utils import property_columns
    elif args['dataset'] == 'alchemy':
        from utils.alchemy_utils import property_columns
    else:
        raise ValueError('The dataset is not supported.')

    molecules = load_data(f'{datadir}molecules.pkl')

    for group_type in ['atomic', 'benson']:
        args["group_type"] = group_type

        print('??????????????????????????')
        print(f'?? Processing {group_type} group')
        print('??????????????????????????')

        for split_type, indices_path in indices_paths.items():
            args["split_type"] = split_type

            print('++++++++++++++++++++++++++')
            print(f'++ Processing {split_type} split')
            print('++++++++++++++++++++++++++')

            # Process and train the linear models
            results = process_and_train(args, datadir, indices_path)
            # Create a copy to prevent modifying the original list
            updated_molecules = molecules.copy()

            # Add the coefficients to the molecules
            for molecule in updated_molecules:
                for prop in results:
                    if group_type == 'atomic':
                        coefs = np.zeros(
                            max(molecule['atomic_groups_ids']) + 1)
                        positions = zip(
                            molecule['atomic_groups_ids'], molecule['atomic_groups'])

                        for pos, group in positions:
                            coefs[pos] = results[prop][group] if group in results[prop] else 0.0

                        molecule[f"{prop}_{group_type}_coefs"] = coefs

                    elif group_type == 'benson':

                        coefs = np.zeros(max(molecule['benson_groups_ids']) + 1)
                        positions = zip(
                            molecule['benson_groups_ids'], molecule['benson_groups'])

                        for pos, group in positions:
                            coefs[pos] = results[prop][group] if group in results[prop] else 0.0

                        molecule[f"{prop}_{group_type}_coefs"] = coefs

                    else:
                        raise ValueError(
                            f"Unsupported group type: {group_type}.")

            # Save the processed molecules separately for each split type
            with open(f'{datadir}molecules_{split_type}_split.pkl', 'wb') as f:
                pickle.dump(updated_molecules, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
