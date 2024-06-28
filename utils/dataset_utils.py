from rdkit import Chem
from itertools import product, combinations_with_replacement

def progress_bar(i, total, length=20):
    """
    Just for tracking the progress of the code.
    """
    percent = (i / total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f"\r|{bar}| {percent*100:.2f}% ", end='')

def progress_bar_file(filename, i, total, length=20):
    """
    Just for tracking the progress of the training.
    """
    percent = (i / total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(f"|{bar}| {percent*100:.2f}%\n")
        
def generate_groups(center_atoms, neighbor_atoms, chemical_data, min_neighbors=2):
    """
    Create all the possible groups considering min of neighbors, and the center 
    atoms and their chemical constraints.
    """
    groups = []
    bonds = {1.0: '-', 1.5: ':', 2.0: '=', 3.0: '#'}
    unique_smarts = set()
    for center in center_atoms:
        # Iterate over combinations of bond types up to the maximum number of bonds
        for n_bonds in range(min_neighbors, chemical_data[center]['max_bonds'] + 1):
            for bond_types in product(chemical_data[center]['bond_types'], repeat=n_bonds):
                # Filter out combinations that does not make sense
                if 1.5 in bond_types and center == "C":
                    if sorted(bond_types) == [1.5, 1.5, 2.0]:
                        pass
                    else:
                        if bond_types.count(1.5) < 2:
                            continue
                        elif not (chemical_data[center]['valency'] <= sum(bond_types) <= 
                                  chemical_data[center]['valency'] + 0.5):
                            continue
                elif 1.5 in bond_types and center == "N":
                    if bond_types.count(1.5) == 3:
                        pass
                    elif not(bond_types.count(1.5) == 2 and 
                             chemical_data[center]['valency'] <= sum(bond_types) <= 
                             chemical_data[center]['valency'] + 1):
                        continue
                elif 1.5 in bond_types and center == "O":
                    if bond_types.count(1.5) < 2:
                        continue
                    if not (chemical_data[center]['valency'] <= sum(bond_types) <= 
                            chemical_data[center]['valency'] + 1):
                        continue
                else:
                    if sum(bond_types) != chemical_data[center]['valency']:
                        continue
                for neighbors in combinations_with_replacement(neighbor_atoms, len(bond_types)):
                    # Check if neighbors can form the required bonds
                    if not all(chemical_data[n]['max_bond_order'] >= b for n, b in zip(neighbors, bond_types)):
                        continue
                    # Sort neighbors and bond_types together based on bond types and then neighbors
                    neighbors_bonds = sorted(zip(neighbors, bond_types), key=lambda x: (x[1], x[0]))
                    sorted_neighbors, sorted_bond_types = zip(*neighbors_bonds)
                    smarts = f"{center}" + ''.join(f"({bonds[b]}{n})" for n, b in zip(sorted_neighbors, sorted_bond_types))
                    # Check for uniqueness
                    if smarts not in unique_smarts:
                        unique_smarts.add(smarts)
                        group = {'smarts': smarts, 
                                 'center_atom': center, 
                                 'neighbors': sorted_neighbors, 
                                 'bonds': sorted_bond_types, 
                                 'n_neighbors': len(sorted_neighbors)}
                        groups.append(group)
    return groups

def match_group(mol, atom, group):
    """
    Get the groups that match an atom in a molecule.
    """
    # Check whether the center atom matches
    if atom.GetSymbol() != group['center_atom']:
        return False
    neighbors = atom.GetNeighbors()
    neighbor_symbols = [n.GetSymbol() for n in neighbors]
    # Check whether the neighbors matches
    if sorted(neighbor_symbols) != sorted(group['neighbors']):
        return False
    bond_types = [mol.GetBondBetweenAtoms(atom.GetIdx(), n.GetIdx()).GetBondTypeAsDouble() for n in neighbors]
    # Combine neighbor symbols and bond types into tuples
    actual_neighbor_bond_pairs = sorted(zip(neighbor_symbols, bond_types))
    # Combine expected neighbor symbols and bond types into tuples
    expected_neighbor_bond_pairs = sorted(zip(group['neighbors'], group['bonds']))
    # Compare the actual and expected neighbor-bond pairs
    return actual_neighbor_bond_pairs == expected_neighbor_bond_pairs
    
def get_groups_data(molecule, mol, benson_groups, atomic_groups):
    """
    Atribute the groups that are present in the molecule
    """
    adj_list_from = []
    adj_list_to = []
    group_mask = [False]*len(molecule['charges'])
    molecule_benson_groups = []
    molecule_atomic_groups = []
    molecule_benson_groups_ids = []
    molecule_atomic_groups_ids = []
    # Loop through all atoms in the molecule
    for atom in mol.GetAtoms():
        benson_matched_groups = [group for group in benson_groups if match_group(mol, atom, group)]
        atomic_matched_groups = [group for group in atomic_groups if match_group(mol, atom, group)]
        if (len(benson_matched_groups) > 1) or (len(atomic_matched_groups) > 1):
            raise ValueError(f"More than one group match found for atom {atom.GetSymbol()} \
                                with neighbors {[n.GetSymbol() for n in atom.GetNeighbors()]}\n\
                                SMILES: {Chem.MolToSmiles(mol)}")
        elif benson_matched_groups:
            group_mask[atom.GetIdx()] = True
            from_list = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
            to_list   = [atom.GetIdx()]*len(from_list)
            adj_list_from.extend(from_list)
            adj_list_from.extend(to_list)
            adj_list_to.extend(to_list)
            adj_list_to.extend(from_list)
            molecule_benson_groups.append(benson_matched_groups[0]['smarts'])
            molecule_benson_groups_ids.append(atom.GetIdx())
        if not atomic_matched_groups:
            raise ValueError(f"No atomic group match found for atom {atom.GetSymbol()} \
                                with neighbors {[n.GetSymbol() for n in atom.GetNeighbors()]}\n\
                                SMILES: {Chem.MolToSmiles(mol)}")
        molecule_atomic_groups.append(atomic_matched_groups[0]['smarts'])
        molecule_atomic_groups_ids.append(atom.GetIdx())
    molecule['benson_groups'] = molecule_benson_groups
    molecule['benson_groups_ids'] = molecule_benson_groups_ids
    molecule['atomic_groups'] = molecule_atomic_groups
    molecule['atomic_groups_ids'] = molecule_atomic_groups_ids
    molecule['group_adj']  = [adj_list_from, adj_list_to]
    molecule['group_mask'] = group_mask
    return molecule 

def count_groups(matched_groups, groups):
    """
    Count the number of each group in a molecule
    """
    group_counts = {group['smarts']: 0 for group in groups}
    # Increment count for each group found
    for group in matched_groups:
        if group in group_counts:
            group_counts[group] += 1
    return group_counts

def generate_scaffold(mol, include_chirality=False):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffold_smiles = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol, includeChirality=include_chirality)
    return scaffold_smiles
    
def scaffold_split(mol_list, splits):
    """
    Split the data based on scaffolds.
    """
    if sum(splits) != 1.0:
        raise ValueError('The values of the splits does not sum 1.0')
    scaffolds = {}
    for id, mol in enumerate(mol_list):
        try:
            scaffold = generate_scaffold(mol)
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [id]
            else:
                scaffolds[scaffold].append(id)
        except Exception as e:
            print(e, flush=True)
            continue
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [scaffold_set for (scaffold, scaffold_set) in 
                     sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]
    total_size = len(mol_list)
    train_end  = int(splits[0] * total_size)
    val_end    = train_end + int(splits[1] * total_size)
    train_ids, valid_ids, test_ids = [], [], []
    for scaffold_set in scaffold_sets:
        if len(train_ids) + len(scaffold_set) > train_end:
            if len(train_ids) + len(scaffold_set) + len(valid_ids) > val_end:
                test_ids += scaffold_set
            else:
                valid_ids += scaffold_set
        else:
            train_ids += scaffold_set
    return train_ids, valid_ids, test_ids