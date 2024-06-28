import torch

def collate_fn(batch):
    """
    Used for preparing the data for batch training.
    """

    batch_data = {}
    
    # Collate each property
    for key in batch[0].keys():

        if key not in ['smiles', 'file', 'benson_groups', 'atomic_groups', 'CHEMBL_ID', 'CONF_ID', 'group_adj']: # Remove non-numerical data
        
            key_data = [torch.tensor(mol[key]) for mol in batch] # Convert the data into tensors

            if key_data[0].dim() == 0: # Collect the scalars here. Stack to guarantee the first dimention is always batch.
                batch_data[key] = torch.stack(key_data)

            elif key_data[0].dim() in [1, 2]: # Collect the positions here.
                batch_data[key] = torch.nn.utils.rnn.pad_sequence(key_data, batch_first=True, padding_value=0)

            else:
                raise(f'Houston, something wrong here (key: {key}). Please check. Problem with the stacking and the padding. ')
            
            if key.endswith('_coefs') and batch_data[key].shape[1] < batch_data['charges'].shape[1]:
                padding_needed = batch_data['charges'].shape[1] - batch_data[key].shape[1]
                batch_data[key] = torch.nn.functional.pad(batch_data[key], (0, padding_needed))

        elif key in ['smiles', 'file', 'benson_groups', 'atomic_groups']:
            key_data = [mol[key] for mol in batch]
            batch_data[key] = key_data
    
    atom_mask = batch_data['charges'] > 0
    batch_data['atom_mask'] = atom_mask

    n_nodes = batch_data['positions'].size(1)

    ajd_list = [mol['group_adj'] for mol in batch]
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
    Adjusts adjacency lists with an offset for each graph in a batch.

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