from .dataset_utils import get_groups_data
import os
import re
import ssl
import shutil
import urllib
import certifi

from datetime import datetime

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename="(.+)"', cd)
    return fname[0] if fname else None


def download_dataset_qmugs(datadir, reportfile):
    """
    Download the QMugs dataset from original source.
    """
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Delete in case the dataset is already there
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
    os.makedirs(datadir)
    context = ssl.create_default_context(cafile=certifi.where())
    print(
        f'Beginning download of QMugs dataset! Output will be in directory: \n{datadir}.')
    url_data = 'https://libdrive.ethz.ch/index.php/s/X5vOBNSITAG5vzM/download?path=%2F&files=structures.tar.gz'

    with urllib.request.urlopen(url_data, context=context) as response:
        cd = response.headers.get('Content-Disposition')
        filename = get_filename_from_cd(cd) or 'default_filename.tar.gz'
        tar_data = os.path.join(datadir, filename)

        with open(tar_data, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

    print('QMugs dataset downloaded successfully!')
    with open(reportfile, 'w', encoding='utf-8') as file:
        file.write(f"\nThe QMugs dataset was downloaded on {date}\n")
        file.write(f"It was extracted from:\n{url_data}\n")


def process_file(datafile, benson_groups, atomic_groups):
    """
    Extract the data from the .sdf file from Alchemy, do processing and save data in a dictionary
    """

    z_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8,
              'F': 9, 'S': 16, 'Cl': 17, 'Br': 35, "I": 53}
    z_list = [1, 6, 7, 8, 9, 16, 17, 35, 53]

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

    properties = mol.GetPropsAsDict()
    property_names = list(mol.GetPropNames())
    for prop in property_names:
        mol.ClearProp(prop)

    prop_strings = ['CHEMBL_ID', 'CONF_ID', 'DFT:TOTAL_ENERGY', 'DFT:ATOMIC_ENERGY', 'DFT:FORMATION_ENERGY',
                    'DFT:HOMO_ENERGY', 'DFT:LUMO_ENERGY', 'DFT:HOMO_LUMO_GAP']

    mol_props = {key: properties[key]
                 for key in prop_strings if key in properties}

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
    hartree_to_eV = {'DFT:TOTAL_ENERGY': 27.2114, 'DFT:ATOMIC_ENERGY': 27.2114, 'DFT:FORMATION_ENERGY': 27.2114,
                     'DFT:HOMO_ENERGY': 27.2114, 'DFT:LUMO_ENERGY': 27.2114, 'DFT:HOMO_LUMO_GAP': 27.2114}
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
    'Cl': {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]},
    'Br': {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]},
    'I':  {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]}
}

property_columns = ['DFT:TOTAL_ENERGY', 'DFT:ATOMIC_ENERGY', 'DFT:FORMATION_ENERGY',
                    'DFT:HOMO_ENERGY', 'DFT:LUMO_ENERGY', 'DFT:HOMO_LUMO_GAP']


center_atoms = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
neighbor_atoms = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
