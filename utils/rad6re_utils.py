import os
import re
import ssl
import shutil
import zipfile
import urllib.request
import certifi

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .dataset_utils import get_groups_data, progress_bar
from datetime import datetime

def clear_directory_contents(dir_path):
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

def download_dataset_hrad(datadir, reportfile):
    """
    Download the ΔHRad-6-RE dataset from original source.
    """

    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Delete in case the dataset is already there
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
    os.makedirs(datadir)

    context = ssl.create_default_context(cafile=certifi.where())
    print(f'Beginning download of ΔHRad-6-RE dataset! Output will be in directory: \n{datadir}.')
    url_data = 'https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-19267-x/MediaObjects/41467_2020_19267_MOESM3_ESM.zip'
    zip_data = os.path.join(datadir, '41467_2020_19267_MOESM3_ESM.zip')
    
    with urllib.request.urlopen(url_data, context=context) as response, open(zip_data, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)

    print('ΔHRad-6-RE dataset downloaded successfully!')
    with open(reportfile, 'w', encoding='utf-8') as file:
        file.write(f"\nThe ΔHRad-6-RE dataset was downloaded on {date}\n")
        file.write(f"It was extracted from:\n{url_data}\n")


def get_datasets(args, benson_groups, atomic_groups):
    """
    Process the files for different datasets.
    """
    try:
        clear_directory_contents(args['datadir'])
        print(f"I've cleanned everything that was in the args['datadir'] before.")
    except Exception as e:
        print(f"I'm downloading the data, but there was nothing to remove here before.")

    download_dataset_hrad(args['datadir'], args['report_file'])
    zip_data = os.path.join(args['datadir'], '41467_2020_19267_MOESM3_ESM.zip')

    data_file = os.path.join(args['datadir'], 'Supplementary_Data_1/Rad-6_databases.xyz')
    reaction_file = os.path.join(args['datadir'], 'Supplementary_Data_1/Rad-6-RE_network.txt')

    print("Extracting the zip file...")
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        zip_ref.extractall(args['datadir'])
    os.remove(zip_data)

    z_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    z_list = [1, 6, 7, 8, 9]

    molecules = []
    errors = []

    with open(data_file) as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        
        progress_bar(i, len(lines))

        num_atoms = int(lines[i].strip())
        i += 1
        mol_props = lines[i].split()
        i += 1
        mol_xyz = lines[i:i + num_atoms]
        i += num_atoms

        atom_charges, atom_positions = [], []
        for line in mol_xyz:
            atom, posx, posy, posz, _, _, _ = line.replace('*^', 'e').split()
            atom_charges.append(z_dict[atom])
            atom_positions.append([float(posx), float(posy), float(posz)])

        properties = {}
        for item in mol_props:
            key, value = item.split('=', 1)  
            if key in ['smile', 'id']:
                value = value.strip('"')
                properties[key] = value
            elif key in ['energy', 'AE', 'BS-DFT']:
                properties[key] = float(value)

        molecule = {'num_atoms': num_atoms,
                    'charges'  : atom_charges, 
                    'positions': atom_positions}
        molecule.update(properties)

        included_species = sorted(set(molecule['charges']))
        if included_species[0] == 0:
            included_species = included_species[1:]

        molecule['one_hot'] = [[int(charge == species) for species in z_list] 
                               for charge in molecule['charges']]
        molecule['num_species'] = len(included_species)
        molecule['max_charge'] = max(included_species)

        try:
            mol = Chem.MolFromSmiles(molecule['smile'])
            mol = Chem.AddHs(mol)
            if mol is not None:
                molecule = get_groups_data(molecule, mol, benson_groups, atomic_groups)
                molecule['mol'] = mol
                molecules.append(molecule)
        except (ValueError, TypeError, RuntimeError) as e:
            errors.append(e)   

    molecules_by_id = {molecule['id']: molecule for molecule in molecules}

    with open(reaction_file) as file:
        lines = file.readlines()

    lines = lines[1:]
    reactions = []

    for line in lines:
        try:
            entries = re.split(r'[,\s]+', line.strip())
            if len(entries) != 4:
                raise ValueError(
                    "Line does not contain exactly four elements.")
            r_id, p1_id, p2_id, RE = entries

            # Check if the IDs exist in the dictionary, raise an exception if not
            if r_id not in molecules_by_id:
                raise ValueError(f"{r_id} not found.")
            if p1_id not in molecules_by_id:
                raise ValueError(f"{p1_id} not found.")

            reactant = molecules_by_id[r_id]
            product_1 = molecules_by_id[p1_id]

            # Check for the special case where p2_id is "[]"
            if p2_id == '"[x]"':
                reaction = {
                    'reactant': [reactant], 
                    'product' : [product_1],
                    'dHrxn': float(RE)}
            else:
                if p2_id not in molecules_by_id:
                    raise ValueError(f"{p2_id} not found.")
                product_2 = molecules_by_id[p2_id]
                reaction = {
                'reactant': [reactant], 
                'product' : [product_1, product_2],
                'dHrxn': float(RE)}            

            reactions.append(reaction)

        except (ValueError, TypeError, RuntimeError) as e:
            errors.append(e)

    shutil.rmtree(os.path.join(args['datadir'], 'Supplementary_Data_1'))
    print("\nDone with the processing...")

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