from utils.dataset_utils import (
    progress_bar,
    progress_bar_file,
    scaffold_split,
    generate_groups,
    count_groups
    # count_groups_in_smiles
)
from argparse import ArgumentParser
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
import seaborn as sns

import random
import zipfile
import tarfile
import shutil
import pickle
import os

import matplotlib.pyplot as plt


def clear_directory_contents(dir_path: str) -> None:
    """
    Remove all files and subdirectories in the specified directory
    while keeping the directory itself intact.
    """
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def safe_extract_with_filter(tar_path: str, extract_to: str, tar_type: str = 'bz2') -> None:
    """
    Safely extracts a tar archive to a specified directory using a filter to
    ensure the paths are safe and to avoid extracting unwanted files.
    """
    def member_filter(member, path):
        if member.islnk() or member.issym():
            return None  # Skip symbolic links
        if ".." in member.name or member.name.startswith("/"):
            return None  # Skip paths that could lead to directory traversal
        return member
    with tarfile.open(tar_path, f'r:{tar_type}') as tar:
        tar.extractall(path=extract_to, members=tar.getmembers(),
                       filter=member_filter)


def get_datasets(
        args: Dict[str, Any],
        benson_groups: List[Dict[str, Any]],
        atomic_groups: List[Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process the files for different datasets.
    """

    if args['dataset'] != 'qmugs':
        try:
            clear_directory_contents(datadir)
            print(f"I've cleanned everything that was in the datadir before.")
        except Exception as e:
            print(f"I'm downloading the data, but there was nothing to remove here before.")

    if args['dataset'] == 'qm9':

        from utils.qm9_utils import (
            convert_units,
            get_thermo_dict,
            process_file,
            add_thermo_targets,
            download_dataset_qm9)

        download_dataset_qm9(datadir, args['report_file'])
        thermo_dict = get_thermo_dict(datadir)
        files_dir = os.path.join(datadir, 'dsgdb9nsd')

        print("Extracting the tar file...")
        safe_extract_with_filter(os.path.join(
            datadir, 'dsgdb9nsd.xyz.tar.bz2'), files_dir)
        os.remove(os.path.join(datadir, 'dsgdb9nsd.xyz.tar.bz2'))
        files = os.listdir(files_dir)
        files = [os.path.join(files_dir, file) for file in files]

    elif args['dataset'] == 'qmugs':

        from utils.qmugs_utils import (
            convert_units,
            process_file)
        
        files_dir = os.path.join(datadir, 'structures')

        if not os.path.isdir(files_dir):
            raise ValueError('Please download and extract the dataset manually.')
        files = os.listdir(files_dir)
        files = random.sample(files, args['max_num_files'])
        files = [os.path.join(files_dir, file, 'conf_00.sdf')
                 for file in files]

    elif args['dataset'] == 'alchemy':

        from utils.alchemy_utils import (
            convert_units,
            process_file,
            download_dataset_alchemy)

        download_dataset_alchemy(datadir, args['report_file'])
        files_dir = os.path.join(datadir, 'Alchemy-v20191129')

        print("Extracting the zip file...")
        with zipfile.ZipFile(os.path.join(datadir, 'alchemy-v20191129.zip'), 'r') as zip_ref:
            zip_ref.extractall(datadir)
        os.remove(os.path.join(datadir, 'alchemy-v20191129.zip'))

        entries = os.listdir(files_dir)
        atom_directories = [
            entry for entry in entries if entry.startswith('atom')]
        files = []

        for atom_dir in atom_directories:
            full_dir_path = os.path.join(files_dir, atom_dir)
            f = os.listdir(full_dir_path)
            files.extend(os.path.join(full_dir_path, x) for x in f)
        data_df = pd.read_csv(f'{files_dir}/final_version.csv')

    else:
        raise ValueError('The dataset is not supported.')

    mols = []
    molecules = []
    benson_rows = []
    atomic_rows = []
    error_files = []
    benson_groups_names = [group["smarts"] for group in benson_groups]
    atomic_groups_names = [group["smarts"] for group in atomic_groups]
    df_benson = pd.DataFrame(columns=benson_groups_names + property_columns)
    df_atomic = pd.DataFrame(columns=atomic_groups_names + property_columns)

    for idx, file in enumerate(files):
        if idx % 100 == 0 or idx == len(files)-1:
            progress_bar(idx, len(files)-1)
            progress_bar_file(args['report_file'], idx, len(files)-1)

        try:  # Process each molecule
            if args['dataset'] == 'alchemy':
                molecule, mol = process_file(
                    file, data_df, benson_groups, atomic_groups)
            else:
                molecule, mol = process_file(
                    file, benson_groups, atomic_groups)

        except Exception as e:
            error_files.append(f'{file} : {e}')
            continue

        # Make the necessary corrections
        if args['thermo_correction'] and args['dataset'] == 'qm9':
            molecule = add_thermo_targets(molecule, thermo_dict)
            thermo_targets = [
                key.split('_')[0] for key in molecule.keys() if key.endswith('_thermo')]
            for key in thermo_targets:
                molecule[key] -= molecule[key + '_thermo']

        if args['convert_ev']:
            molecule = convert_units(molecule)

        # Create the rows for the dataframes for linear regression
        row_benson = {prop: molecule[prop]
                      for prop in df_benson.columns if prop in molecule}
        benson_group_counts = count_groups(
            molecule['benson_groups'], benson_groups)

        row_atomic = {prop: molecule[prop]
                      for prop in df_atomic.columns if prop in molecule}
        atomic_group_counts = count_groups(
            molecule['atomic_groups'], atomic_groups)

        for group in benson_groups_names:
            row_benson[group] = benson_group_counts[group]
        for group in atomic_groups_names:
            row_atomic[group] = atomic_group_counts[group]

        # Append all the data generated
        molecules.append(molecule)
        mols.append(mol)
        benson_rows.append(row_benson)
        atomic_rows.append(row_atomic)

    # Merge the rows into the dataframes
    df_benson = pd.concat(
        [df_benson, pd.DataFrame(benson_rows)], ignore_index=True)
    df_atomic = pd.concat(
        [df_atomic, pd.DataFrame(atomic_rows)], ignore_index=True)

    # Remove the extracted files, as they are no longer needed
    shutil.rmtree(files_dir)
    print("\nDone with the processing...")

    return df_benson, df_atomic, molecules, mols, error_files


if __name__ == "__main__":

    parser = ArgumentParser(description="Arguments to run the experiment.")
    parser.add_argument("--dataset",           default='qm9')
    parser.add_argument("--download",          default=True)
    parser.add_argument("--max_num_files",     default=200000, type=int)
    parser.add_argument("--thermo_correction", default=True)
    parser.add_argument("--convert_ev",        default=True)
    parser.add_argument("--splits",            default=[0.7, 0.15, 0.15])
    parser.add_argument("--datadir_root",      default='./datasets/')
    parser.add_argument("--report_file_root",  default='./reports/')
    args, unknown = parser.parse_known_args()
    args = vars(args)
    args['report_file'] = args['report_file_root'] + \
        args['dataset'] + '_report.txt'

    datadir = args['datadir_root'] + args['dataset'] + '/data/'

    if sum(args['splits']) != 1.0:  # Check the splits parameters
        raise ValueError('The values of the splits does not sum 1.0')

    # Gather the necessary data based on the dataset
    if args['dataset'] == 'qm9':
        from utils.qm9_utils import chemical_data, property_columns, center_atoms, neighbor_atoms
    elif args['dataset'] == 'qmugs':
        from utils.qmugs_utils import chemical_data, property_columns, center_atoms, neighbor_atoms
    elif args['dataset'] == 'alchemy':
        from utils.alchemy_utils import chemical_data, property_columns, center_atoms, neighbor_atoms
    else:
        raise ValueError('The dataset is not supported.')

    # Generate the groups
    benson_groups = generate_groups(
        center_atoms, neighbor_atoms, chemical_data)
    atomic_groups = generate_groups(
        center_atoms, neighbor_atoms, chemical_data, 1)

    # Process the dataset
    benson_df, atomic_df, molecules, mols, error_files = get_datasets(
        args, benson_groups, atomic_groups)

    # Report relevant information
    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"The number of process molecules was {len(molecules)}\n")
        file.write(f"The number of error files was {len(error_files)}\n")

    # Save all the processed data
    with open(f'{datadir}/atomic_groups.pkl', 'wb') as f:
        pickle.dump(atomic_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/benson_groups.pkl', 'wb') as f:
        pickle.dump(benson_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/molecules.pkl', 'wb') as f:
        pickle.dump(molecules, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/mols.pkl', 'wb') as f:
        pickle.dump(mols, f, protocol=pickle.HIGHEST_PROTOCOL)
    benson_df.to_csv(f'{datadir}/data_benson_groups.csv', index=False)
    atomic_df.to_csv(f'{datadir}/data_atomic_groups.csv', index=False)

    # Create random splits
    total_size = len(molecules)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    print(f'This is a sample of the training indices\n{indices[:10]}')
    print('Please check whether the values are random.')
    train_end = int(args['splits'][0] * total_size)
    val_end = train_end + int(args['splits'][1] * total_size)
    train_indices_random = indices[:train_end]
    val_indices_random = indices[train_end:val_end]
    test_indices_random = indices[val_end:]

    # Create scaffold splits
    train_indices_scaffold, val_indices_scaffold, test_indices_scaffold = scaffold_split(
        mols, args['splits'])

    # Save the indices from the splits
    with open(f'{datadir}/train_indices_random.pkl', 'wb') as f:
        pickle.dump(train_indices_random.tolist(), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/val_indices_random.pkl', 'wb') as f:
        pickle.dump(val_indices_random.tolist(), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/test_indices_random.pkl', 'wb') as f:
        pickle.dump(test_indices_random.tolist(), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/train_indices_scaffold.pkl', 'wb') as f:
        pickle.dump(train_indices_scaffold, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/val_indices_scaffold.pkl', 'wb') as f:
        pickle.dump(val_indices_scaffold, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{datadir}/test_indices_scaffold.pkl', 'wb') as f:
        pickle.dump(test_indices_scaffold, f, protocol=pickle.HIGHEST_PROTOCOL)
