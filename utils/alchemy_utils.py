from datetime import datetime
from .dataset_utils import get_groups_data
import os
import ssl
import shutil
import urllib
import certifi

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def download_dataset_alchemy(datadir, reportfile):
    """
    Download the Achemy dataset from original source.
    """
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Delete in case the dataset is already there
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
    os.makedirs(datadir)
    context = ssl.create_default_context(cafile=certifi.where())
    print(
        f'Beginning download of Alchemy dataset! Output will be in directory: \n{datadir}.')
    url_data = 'https://alchemy.tencent.com/data/alchemy-v20191129.zip'
    tar_data = os.path.join(datadir, 'alchemy-v20191129.zip')
    with urllib.request.urlopen(url_data, context=context) as response, open(tar_data, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print('Alchemy dataset downloaded successfully!')
    with open(reportfile, 'w', encoding='utf-8') as file:
        file.write(f"\nThe Alchemy dataset was downloaded on {date}\n")
        file.write(f"It was extracted from:\n{url_data}\n")


def process_file(datafile, data_df, benson_groups, atomic_groups):
    """
    Extract the data from the .sdf file from Alchemy, 
    do processing and save data in a dictionary
    """

    z_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'S': 16, 'Cl': 17}
    z_list = [1, 6, 7, 8, 9, 16, 17]

    suppl = Chem.SDMolSupplier(datafile, removeHs=False)
    mol = next(suppl)

    atom_charges, atom_positions = [], []
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atom_positions.append([float(pos.x), float(pos.y), float(pos.z)])
        atom_charges.append(z_dict[atom.GetSymbol()])
    num_atoms = mol.GetNumAtoms()

    mol_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    mol_smiles = Chem.CanonSmiles(mol_smiles)

    tag = os.path.basename(datafile)[:-4]

    row = data_df.loc[data_df['gdb_idx'] == int(tag)]
    mol_props = row.iloc[0].values.tolist()

    prop_strings = ['gdb_idx', 'atom_number', 'zpve', 'Cv', 'gap',
                    'G', 'homo', 'U', 'alpha', 'U0', 'H', 'lumo', 'mu', 'r2']
    mol_props = [int(x) for x in mol_props[:2]] + [float(x)
                                                   for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges,
                'positions': atom_positions, 'smiles': mol_smiles, 'file': os.path.basename(datafile)}
    molecule.update(mol_props)

    included_species = sorted(set(molecule['charges']))
    if included_species[0] == 0:
        included_species = included_species[1:]

    molecule['one_hot'] = [[int(charge == species) for species in z_list]
                           for charge in molecule['charges']]

    molecule['num_species'] = len(included_species)
    molecule['max_charge'] = max(included_species)

    molecule = get_groups_data(molecule, mol, benson_groups, atomic_groups)

    return molecule, mol


def convert_units(data):
    hartree_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114,
                     'zpve': 27.2114, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
    for key in data.keys():
        if key in hartree_to_eV:
            data[key] *= hartree_to_eV[key]
    return data


chemical_data = {
    'H':  {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]},
    'C':  {'valency': 4, 'max_bonds': 4, 'max_bond_order': 3, 'bond_types': [1.0, 1.5, 2.0, 3.0]},
    'O':  {'valency': 2, 'max_bonds': 2, 'max_bond_order': 2, 'bond_types': [1.0, 1.5, 2.0]},
    'N':  {'valency': 3, 'max_bonds': 3, 'max_bond_order': 3, 'bond_types': [1.0, 1.5, 2.0, 3.0]},
    'F':  {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]},
    'S':  {'valency': 2, 'max_bonds': 2, 'max_bond_order': 2, 'bond_types': [1.0, 1.5, 2.0]},
    'Cl': {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]}
}

property_columns = ['mu', 'alpha', 'homo', 'lumo',
                    'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
center_atoms = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
neighbor_atoms = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
