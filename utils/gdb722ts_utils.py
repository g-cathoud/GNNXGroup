from datetime import datetime
from .dataset_utils import get_groups_data, progress_bar

import os

import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def list_files(xyz_files_dir):

    items = os.listdir(xyz_files_dir)
    molecules_files = []

    for i in items:
        if os.path.isdir(os.path.join(xyz_files_dir, i)):

            if len(os.listdir(os.path.join(xyz_files_dir, i))) == 3:
                files = {}
                for file in os.listdir(os.path.join(xyz_files_dir, i)):
                    if file.startswith('p'):
                        files['p'] = os.path.join(xyz_files_dir, i, file)
                    elif file.startswith('r'):
                        files['r'] = os.path.join(xyz_files_dir, i, file)
                molecules_files.append(files)

    return molecules_files


def filter_files(df, files):

    charged_molecules = []
    filtered_files = []
    errors = []

    for file in files:
        try:
            idx = int(os.path.basename(file['r'])[1:-4])

            r_smile = df.loc[df['idx'] == idx, 'rsmi'].values[0]
            dE0 = df.loc[df['idx'] == idx, 'dE0'].values[0]
            dHrxn = df.loc[df['idx'] == idx, 'dHrxn298'].values[0]
            r_mol = Chem.MolFromSmiles(r_smile)
            r_mol = Chem.AddHs(r_mol)

            r_count = 0
            for atom in r_mol.GetAtoms():
                if atom.GetFormalCharge() != 0:
                    r_count += 1

            p_smile = df.loc[df['idx'] == idx, 'psmi'].values[0]
            p_mol = Chem.MolFromSmiles(p_smile)
            p_mol = Chem.AddHs(p_mol)

            p_count = 0
            for atom in p_mol.GetAtoms():
                if atom.GetFormalCharge() != 0:
                    p_count += 1
            if p_count > 0 or r_count > 0:
                charged_molecules.append(os.path.basename(file['r'])[1:-4])
            else:
                file['r_smile'] = r_smile
                file['p_smile'] = p_smile
                file['dE0'] = dE0
                file['dHrxn'] = dHrxn
                filtered_files.append(file)
        except:
            errors.append(os.path.basename(file['r'])[1:-4])

    return filtered_files, errors


def process_file(datafile, smiles, benson_groups, atomic_groups):
    """
    Extract the data from the .xyz file from QM9, do processing and save data in a dictionary
    """

    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    atom_map_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
    sorted_atom_map = sorted(atom_map_to_idx.items())
    mapping_idxs = [idx for _, idx in sorted_atom_map]

    mol = Chem.RenumberAtoms(mol, mapping_idxs)

    rdkit_positions = [atom.GetIdx() + 1 for atom in mol.GetAtoms()]
    map_positions = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    rdkit_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    z_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    z_list = [1, 6, 7, 8, 9]

    xyz_lines = open(datafile).readlines()

    num_atoms = int(xyz_lines[0])
    mol_xyz = xyz_lines[2:num_atoms+2]

    atom_charges, atom_positions, atoms_symbols = [], [], []
    for line in mol_xyz:
        atom, posx, posy, posz = line.replace('*^', 'e').split()
        atom_charges.append(z_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])
        atoms_symbols.append(atom)

    if rdkit_symbols != atoms_symbols:
        raise ValueError(f'Atoms in SMILES and .xyz file do not match: \n{rdkit_symbols} \n{atoms_symbols} \n({smiles})')
    
    if rdkit_positions != map_positions:
        raise ValueError(
            f'Atoms in SMILES and .xyz file do not match: \n{rdkit_positions} \n{map_positions}')

    molecule = {'num_atoms': num_atoms,
                'charges': atom_charges,
                'positions': atom_positions,
                'smile': smiles,
                'file': os.path.basename(datafile)}

    included_species = sorted(set(molecule['charges']))
    if included_species[0] == 0:
        included_species = included_species[1:]

    molecule['one_hot'] = [[int(charge == species) for species in z_list]
                           for charge in molecule['charges']]

    molecule['num_species'] = len(included_species)
    molecule['max_charge'] = max(included_species)

    molecule = get_groups_data(molecule, mol, benson_groups, atomic_groups)

    molecule['mol'] = mol

    return molecule

def get_datasets(args, benson_groups, atomic_groups):

    xyz_files_dir = os.path.join(args['datadir_temp'], 'xyz/')
    files = list_files(xyz_files_dir)

    df = pd.read_csv(os.path.join(args['datadir_temp'], 'gdb.csv'))

    filtered_files, errors = filter_files(df, files)

    reactions = []
    for idx, file in enumerate(filtered_files):

        progress_bar(idx, len(filtered_files))

        try:
            reactant = process_file(
                file['r'], file['r_smile'], benson_groups, atomic_groups)
            product = process_file(
                file['p'], file['p_smile'], benson_groups, atomic_groups)

            reactions.append({'reactant': [reactant], 
                              'product': [product],
                              'dE0': file['dE0'], 
                              'dHrxn': file['dHrxn']})

        except:
            errors.append(os.path.basename(file['r'])[1:-4])

    return reactions, errors


chemical_data = {
    'H':  {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]},
    'C':  {'valency': 4, 'max_bonds': 4, 'max_bond_order': 3, 'bond_types': [1.0, 1.5, 2.0, 3.0]},
    'O':  {'valency': 2, 'max_bonds': 2, 'max_bond_order': 2, 'bond_types': [1.0, 1.5, 2.0]},
    'N':  {'valency': 3, 'max_bonds': 3, 'max_bond_order': 3, 'bond_types': [1.0, 1.5, 2.0, 3.0]},
    'F':  {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]}
}

center_atoms = ['H', 'C', 'N', 'O', 'F']
neighbor_atoms = ['H', 'C', 'N', 'O', 'F']
