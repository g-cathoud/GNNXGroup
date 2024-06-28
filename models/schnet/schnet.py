import torch
from torch import nn

from .utils import *

"""
This model was extracted from:
https://github.com/atomistic-machine-learning/SchNet
"""

class SchNet(nn.Module):
    def __init__(self,
                 n_hf,
                 n_radial_basis,
                 cutoff,
                 n_interactions,
                 device,
                 shared_interactions=False,
                 activation=shifted_softplus,
                 max_z=100,
                 **kwargs):

        super(SchNet, self).__init__()

        self.n_hf = n_hf
        self.n_radial_basis = n_radial_basis
        self.cutoff = cutoff
        self.n_interactions = n_interactions
        self.device = device
        self.shared_interactions = shared_interactions
        self.activation = activation
        self.max_z = max_z

        self.radial_basis = GaussianRBF(
            n_rbf=self.n_radial_basis, cutoff=self.cutoff)
        self.cutoff_fn = CosineCutoff(self.cutoff)

        # Encoder
        self.embedding = nn.Embedding(self.max_z, self.n_hf, padding_idx=0)

        # SchNet layers
        if self.shared_interactions:
            self.layers = nn.ModuleList([SchNetLayer(
                n_hf=self.n_hf,
                n_rbf=self.radial_basis.n_rbf,
                activation=self.activation,
            )] * self.n_interactions)
        else:
            self.layers = nn.ModuleList([SchNetLayer(
                n_hf=self.n_hf,
                n_rbf=self.radial_basis.n_rbf,
                activation=self.activation,
            ) for i in range(self.n_interactions)])

        # Read out part
        self.decoder = build_mlp(n_in=self.n_hf, n_out=1)

        self.reset_parameters()
        self.to(self.device)

    def getDistances(self, edge_index, coord):

        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.norm(coord_diff, dim=1)

        return radial.unsqueeze(-1)

    def forward(self, inputs):

        z, coord, edge_index, n_nodes, atom_mask, edge_mask, _, _ = inputs

        radial = self.getDistances(edge_index, coord)

        # compute atom and pair features
        h = self.embedding(z)

        rbf = self.radial_basis(radial.squeeze(-1))
        rbf = rbf * edge_mask

        dist_cut = self.cutoff_fn(radial)
        dist_cut = dist_cut * edge_mask

        row, col = edge_index

        # compute interaction block to update atomic embeddings
        for layer in self.layers:
            h = layer(h, rbf, dist_cut, row, col)

        h = h * atom_mask # Extract only the information that matters

        h = h.view(-1, n_nodes, self.n_hf) # Correct dimentions after batch stacking

        h = torch.sum(h, dim=1) # This summs the embedings of all the nodes into just one.

        pred = self.decoder(h) # Get prediction

        return pred.squeeze(-1), None

    # Linear parameters will be initilized with kaiming_uniform method
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.decoder.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()


class SchNet_atomic(SchNet):
    def __init__(self, *args, **kwargs):
        super(SchNet_atomic, self).__init__(*args, **kwargs)

    def forward(self, inputs):

        z, coord, edge_index, n_nodes, atom_mask, edge_mask, _, _ = inputs

        radial = self.getDistances(edge_index, coord)

        # compute atom and pair features
        h = self.embedding(z)

        rbf = self.radial_basis(radial.squeeze(-1))
        rbf = rbf * edge_mask

        dist_cut = self.cutoff_fn(radial)
        dist_cut = dist_cut * edge_mask

        row, col = edge_index

        # compute interaction block to update atomic embeddings
        for layer in self.layers:
            h = layer(h, rbf, dist_cut, row, col)

        h = h * atom_mask # Extract only the information that matters

        ac = self.decoder(h)  # Extract the atomic contributions

        ac = ac * atom_mask # Extract only the information that matters

        ac = ac.view(-1, n_nodes)  # Correct dimentions after batch stacking

        pred = torch.sum(ac, dim=1) # This sums the atomic contributions to get the final prediction

        return pred, ac


class SchNet_group(SchNet):
    def __init__(self, *args, **kwargs):
        super(SchNet_group, self).__init__(*args, **kwargs)

    def forward(self, inputs):

        z, coord, edge_index, n_nodes, atom_mask, edge_mask, group_adj, group_mask = inputs

        radial = self.getDistances(edge_index, coord)

        # compute atom and pair features
        h = self.embedding(z)

        rbf = self.radial_basis(radial.squeeze(-1))
        rbf = rbf * edge_mask

        dist_cut = self.cutoff_fn(radial)
        dist_cut = dist_cut * edge_mask

        row, col = edge_index

        # compute interaction block to update atomic embeddings
        for layer in self.layers:
            h = layer(h, rbf, dist_cut, row, col)

        h = h * atom_mask # Extract only the information that matters

        hc = unsorted_segment_sum(
            h[group_adj[0]], group_adj[1], num_segments=h.size(0))

        gc = self.decoder(hc)  # Extract the group contributions

        gc = gc*group_mask # Extract only the information that matters

        gc = gc.view(-1, n_nodes)  # Correct dimentions after batch stacking

        pred = torch.sum(gc, dim=1) # This sums the group contributions to get the final prediction

        return pred, gc


class SchNetLayer(nn.Module):
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(self,
                 n_hf,
                 n_rbf,
                 activation=shifted_softplus,
                 ):

        super(SchNetLayer, self).__init__()

        self.in2f = Dense(n_hf, n_hf, bias=False, activation=None)

        self.f2out = nn.Sequential(
            Dense(n_hf, n_hf, activation=activation),
            Dense(n_hf, n_hf, activation=None),
        )

        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_hf, activation=activation), Dense(n_hf, n_hf)
        )

    def forward(self, x, rbf, dist_cut, idx_i, idx_j):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        """

        # Message
        x = self.in2f(x)
        Wij = self.filter_network(rbf)
        Wij = Wij * dist_cut
        x_j = x[idx_j]
        x_ij = x_j * Wij

        # Update
        m = unsorted_segment_sum(x_ij, idx_i, x.shape[0])
        m = self.f2out(m)
        x = x + m

        return x

    # Linear parameters will be initilized with kaiming_uniform method
    def reset_parameters(self):
        for layer in self.in2f.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()
        for layer in self.f2out.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()
        for layer in self.filter_network.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()


def prepare_inputs_schnet(data, input_args):

    batch_size, n_nodes, _ = data['positions'].size()

    atom_positions = data['positions'].view(
        batch_size * n_nodes, -1).to(input_args['device'], input_args['dtype'])
    z = data['charges'].view(batch_size * n_nodes, -
                             1).squeeze(-1).to(input_args['device'], torch.int)

    bond_edges = get_adj_matrix(n_nodes, batch_size)
    bond_edges = bond_edges.to(input_args['device'])

    atom_mask = data['atom_mask'].view(
        batch_size * n_nodes, -1).to(input_args['device'], input_args['dtype'])
    edge_mask = get_edge_mask(data['atom_mask'])
    edge_mask = edge_mask.view(
        batch_size * n_nodes * n_nodes, 1).to(input_args['device'], input_args['dtype'])

    group_mask = data['group_mask'].view(
        batch_size * n_nodes, -1).to(input_args['device'], input_args['dtype'])
    group_adj = data['group_adj'].to(input_args['device'])

    return z, atom_positions, bond_edges, n_nodes, atom_mask, edge_mask, group_adj, group_mask


edges_dic = {}
def get_adj_matrix(n_nodes, batch_size):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size)

    return torch.tensor([rows, cols], dtype=torch.long)


def get_edge_mask(atom_mask):
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    return edge_mask


def get_schnet_model(args):
    kwargs = {
        "n_hf": 128,
        "n_radial_basis": 30,
        "cutoff": 10.0,
        "n_interactions": 7,
        "device": 'cpu',
        "shared_interactions": False,
        "activation": shifted_softplus,
        "max_z": 100}

    for k, v in kwargs.items():
        args.setdefault(k, v)

    # Determine model type
    model_class = {"original": SchNet, "atomic": SchNet_atomic,
                   "group": SchNet_group}.get(args["model_type"])
    if not model_class:
        raise ValueError("Invalid model type specified.")

    model = model_class(**args)
    return model, args
