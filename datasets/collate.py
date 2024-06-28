import torch

def collate_fn(batch):

    batch_data = {}

    reaction_ordered_molecules = []
    reaction_indexes = []
    reaction_indexes_signs = []
    reaction_enthalpies = []

    for idx, reaction in enumerate(batch):

        reactant = reaction['reactant']
        product  = reaction['product']
        reaction_enthalpies.append(reaction['dHrxn'])
        molecules = reactant + product

        reaction_ordered_molecules.extend(molecules)
        reaction_indexes.extend([idx]*len(molecules))
        reaction_indexes_signs.extend([-1]*len(reactant) + [1]*len(product))

    batch_data['n_reactions'] = torch.tensor(len(batch))
    batch_data['reaction_indexes'] = torch.tensor(reaction_indexes)
    batch_data['reaction_indexes_signs'] = torch.tensor(reaction_indexes_signs)
    batch_data['reaction_enthalpy'] = torch.tensor(reaction_enthalpies)

    # Collate each property
    for key in reaction_ordered_molecules[0].keys():

        if key not in ['smile', 'benson_groups', 'atomic_groups', 'group_adj', 'id', 'mol', 'atoms', 'file']:

            # Convert the data into tensors
            key_data = [torch.tensor(mol[key]) for mol in reaction_ordered_molecules]

            # Collect the scalars here. Stack to guarantee the first dimention is always batch.
            if key_data[0].dim() == 0:
                batch_data[key] = torch.stack(key_data)

            elif key_data[0].dim() in [1, 2]:  # Collect the positions here.
                batch_data[key] = torch.nn.utils.rnn.pad_sequence(
                    key_data, batch_first=True, padding_value=0)

            else:
                raise (
                    f'Houston, something wrong here (key: {key}). Please check. Problem with the stacking and the padding. ')

            if key.endswith('_coefs') and batch_data[key].shape[1] < batch_data['charges'].shape[1]:
                padding_needed = batch_data['charges'].shape[1] - \
                    batch_data[key].shape[1]
                batch_data[key] = torch.nn.functional.pad(
                    batch_data[key], (0, padding_needed))

        elif key in ['smile', 'id', 'benson_groups', 'atomic_groups', 'file']:
            key_data = [mol[key] for mol in reaction_ordered_molecules]
            batch_data[key] = key_data

    atom_mask = batch_data['charges'] > 0
    batch_data['atom_mask'] = atom_mask

    n_nodes = batch_data['positions'].size(1)

    ajd_list = [mol['group_adj'] for mol in reaction_ordered_molecules]
    batch_data['group_adj'] = adjust_adj_lists_for_batching(ajd_list, n_nodes)

    batch = batch_data

    return batch


def generate_adjacency_tensor(batch_size, max_nodes):
    """
    Generate an adjacency tensor for undirected, fully connected graphs in batch training.

    Args:
    batch_size (int): The size of the batch.
    max_nodes (int): The maximum number of nodes in the graphs.
    node_masks (torch.Tensor): A binary mask indicating the presence of nodes in each graph.

    Returns:
    torch.Tensor: An adjacency tensor representing the edges of the graphs.
    """
    from_list = []
    to_list = []

    for batch_index in range(batch_size):
        offset = batch_index * max_nodes
        for i in range(max_nodes):
            for j in range(i + 1, max_nodes):
                from_list.extend([offset + i, offset + j])
                to_list.extend([offset + j, offset + i])

    edge_tensor = torch.tensor([from_list, to_list], dtype=torch.long)
    edge_tensor = edge_tensor.view(2, -1)

    return edge_tensor

def adjust_adj_lists_for_batching(all_lists, max_nodes):
    """
    Adjusts adjacency lists for each graph in a batch.

    Parameters:
    all_lists (list of lists): List containing adjacency lists for each graph.
    max_nodes (int): Maximum number of nodes in any graph in the batch.

    Returns:
    list of lists: Adjusted adjacency lists with node indices shifted for batching.
    """

    adjusted_rows = []
    adjusted_cols = []

    for batch_index, adj_list in enumerate(all_lists):
        offset = batch_index * max_nodes
        row, col = adj_list
        adjusted_rows.extend([offset + node for node in row])
        adjusted_cols.extend([offset + node for node in col])

    return torch.tensor([adjusted_rows, adjusted_cols], dtype=torch.long)
