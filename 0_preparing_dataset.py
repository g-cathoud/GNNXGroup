from utils.dataset_utils import generate_groups

from argparse import ArgumentParser

import numpy as np

import pickle
import os


if __name__ == "__main__":

    parser = ArgumentParser(description="Arguments to run the experiment.")
    parser.add_argument("--splits",    default=[0.75, 0.15, 0.10])
    parser.add_argument("--dataset",   default='rad6re')
    parser.add_argument("--datadir",   default='./datasets/')
    parser.add_argument("--reportdir", default='./reports/')

    args, unknown = parser.parse_known_args()

    args = vars(args)
    args['datadir_temp'] = os.path.join(args['datadir'], args['dataset'],'temp') 
    args['datadir'] = os.path.join(args['datadir'], args['dataset'],'data') 
    args['report_file'] = os.path.join(args['reportdir'], f'{args["dataset"]}_dataset.txt' )

    if sum(args['splits']) != 1.0:
        raise ValueError('The values of the splits does not sum 1.0')

    if args['dataset'] == 'rad6re':
        from utils.rad6re_utils import chemical_data, center_atoms, neighbor_atoms, get_datasets
    elif args['dataset'] == 'gdb722ts':
        from utils.gdb722ts_utils import chemical_data, center_atoms, neighbor_atoms, get_datasets

    benson_groups = generate_groups(
        center_atoms, neighbor_atoms, chemical_data)
    atomic_groups = generate_groups(
        center_atoms, neighbor_atoms, chemical_data, 1)

    reactions, errors = get_datasets(args, benson_groups, atomic_groups)

    with open(args['report_file'], 'a', encoding='utf-8') as file:
        file.write(f"The number of processed reactions was {len(reactions)}\n")
        file.write(f"The number of errors was {len(errors)}\n")
    # Saving all the processed data
    with open(f'{args["datadir"]}/atomic_groups.pkl', 'wb') as f:
        pickle.dump(atomic_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{args["datadir"]}/benson_groups.pkl', 'wb') as f:
        pickle.dump(benson_groups, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{args["datadir"]}/reactions.pkl', 'wb') as f:
        pickle.dump(reactions, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Random splits
    total_size = len(reactions)
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    print(f'This is a sample of the training indices\n{indices[:10]}')

    train_end = int(args['splits'][0] * total_size)
    val_end = train_end + int(args['splits'][1] * total_size)
    train_indices_random = indices[:train_end]
    val_indices_random = indices[train_end:val_end]
    test_indices_random = indices[val_end:]

    with open(f'{args["datadir"]}/train_indices_random.pkl', 'wb') as f:
        pickle.dump(train_indices_random.tolist(), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{args["datadir"]}/val_indices_random.pkl', 'wb') as f:
        pickle.dump(val_indices_random.tolist(), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{args["datadir"]}/test_indices_random.pkl', 'wb') as f:
        pickle.dump(test_indices_random.tolist(), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
