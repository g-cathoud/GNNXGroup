import os
import ssl
import shutil
import urllib
import certifi
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from .dataset_utils import get_groups_data
from datetime import datetime

def download_dataset_qm9(datadir, reportfile):
    """
    Download the QM9 (GDB9) dataset from original source.
    """
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Delete in case the dataset is already there
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
    os.makedirs(datadir)
    context = ssl.create_default_context(cafile=certifi.where())
    print(f'Beginning download of GDB9 dataset! Output will be in directory: \n{datadir}.')
    gdb9_url_data = 'https://springernature.figshare.com/ndownloader/files/3195389'
    gdb9_tar_data = os.path.join(datadir, 'dsgdb9nsd.xyz.tar.bz2')
    with urllib.request.urlopen(gdb9_url_data, context=context) as response, open(gdb9_tar_data, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print('GDB9 dataset downloaded successfully!')
    with open(reportfile, 'w', encoding='utf-8') as file:
        file.write(f"\nThe QM9 dataset was downloaded on {date}\n")
        file.write(f"It was extracted from:\n{gdb9_url_data}\n")

def process_file(datafile, benson_groups, atomic_groups):
    """
    Extract the data from the .xyz file from QM9, 
    do processing and save data in a dictionary
    """
    z_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    z_list = [1, 6, 7, 8, 9]

    xyz_lines = open(datafile).readlines()

    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]
    mol_smiles, relax_smiles = xyz_lines[num_atoms+3].split()

    relax_smiles = Chem.CanonSmiles(relax_smiles)
    mol_smiles = Chem.CanonSmiles(mol_smiles)

    xyz_block = f"{num_atoms}\n\n"

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(z_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])
        xyz_block += f"{atom}\t{posx}\t{posy}\t{posz}\n"

    mol = Chem.MolFromXYZBlock(xyz_block)
    if mol is None:
        raise Exception("Problem loading the xyz block")
    rdDetermineBonds.DetermineBonds(mol,charge=0)

    # remove double bond stereo:
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        elif bond.GetBondType() == Chem.BondType.SINGLE:
            bond.SetBondDir(Chem.BondDir.NONE)
    for atom in mol.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    xyz_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
    xyz_smiles = Chem.CanonSmiles(xyz_smiles)

    if xyz_smiles != mol_smiles and xyz_smiles != relax_smiles:
        raise Exception("Problem in the smiles")

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions, 'smiles': xyz_smiles, 'file': os.path.basename(datafile)}
    molecule.update(mol_props)

    included_species = sorted(set(molecule['charges']))
    if included_species[0] == 0:
        included_species = included_species[1:]

    molecule['one_hot'] = [[int(charge == species) for species in z_list] for charge in molecule['charges']]

    molecule['num_species'] = len(included_species)
    molecule['max_charge'] = max(included_species)

    molecule = get_groups_data(molecule, mol, benson_groups, atomic_groups)

    return molecule, mol
   
def convert_units(data, thermo_correction=True):
    if thermo_correction:
        hartree_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27.2114, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114, 'zpve_thermo': 27211.4, 'U0_thermo': 27.2114, 'U_thermo': 27.2114, 'H_thermo': 27.2114,'G_thermo': 27.2114}
    else:
        hartree_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27.2114, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
    for key in data.keys():
        if key in hartree_to_eV:
            data[key] *= hartree_to_eV[key]
    return data

def get_thermo_dict(gdb9dir):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.
    """
    # Download thermochemical energy
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = os.path.join(gdb9dir, 'atomref.txt')
    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()
            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue
            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]] = float(split_therm)
    return therm_energy

def get_unique_charges(charges):
    """
    Get count of each unique charge for each molecule using NumPy.
    """
    charges_np = np.array(charges)
    unique_charges, counts = np.unique(charges_np, return_counts=True)
    charge_counts = {int(z): int(count) for z, count in zip(unique_charges, counts)}
    return charge_counts

def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])
    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = 0
        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z
        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo
    return data

chemical_data = {
    'H': {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]},
    'C': {'valency': 4, 'max_bonds': 4, 'max_bond_order': 3, 'bond_types': [1.0, 1.5, 2.0, 3.0]},
    'O': {'valency': 2, 'max_bonds': 2, 'max_bond_order': 2, 'bond_types': [1.0, 1.5, 2.0]},
    'N': {'valency': 3, 'max_bonds': 3, 'max_bond_order': 3, 'bond_types': [1.0, 1.5, 2.0, 3.0]},
    'F': {'valency': 1, 'max_bonds': 1, 'max_bond_order': 1, 'bond_types': [1.0]}    
}

property_columns = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
center_atoms     = ['H', 'C', 'O', 'N', 'F']
neighbor_atoms   = ['H', 'C', 'O', 'N', 'F']