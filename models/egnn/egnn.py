import torch
from torch import nn

"""
This model was extracted from:
https://github.com/vgsatorras/egnn
"""

class EGNN_var1(nn.Module):
    def __init__(self, 
                 in_node_nf, 
                 in_edge_nf, 
                 hidden_nf, 
                 node_attr=1,
                 device='cpu', 
                 act_fn=nn.SiLU(),
                 n_layers=4, 
                 attention=False, 
                 **kwargs):

        super(EGNN_var1, self).__init__()

        self.in_node_nf = in_node_nf 
        self.in_edge_nf = in_edge_nf
        self.hidden_nf  = hidden_nf
        self.node_attr  = node_attr
        self.device     = device 
        self.act_fn     = act_fn
        self.n_layers   = n_layers
        self.attention  = attention

        if node_attr:
            self.nodes_att_dim = in_node_nf
        else:
            self.nodes_att_dim = 0

        # Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)

        # EGNN layers
        layers = []
        for i in range(0, self.n_layers):
            layers.append(E_GCL(input_nf      = self.hidden_nf,
                                hidden_nf     = self.hidden_nf,  
                                edges_in_d    = self.in_edge_nf, 
                                nodes_att_dim = self.nodes_att_dim, 
                                act_fn        = self.act_fn, 
                                attention     = self.attention))
        self.layers = nn.ModuleList(layers)

        # Read out part
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        
        self.reset_parameters()
        self.to(self.device)
        
    def forward(self, inputs):

        h0, positions, edges, n_nodes, node_mask, edge_mask, n_reactions, reaction_indexes, reaction_indexes_signs, _, _ = inputs
        
        h = self.embedding(h0)

        for layer in self.layers:
            if self.node_attr:
                h, _ = layer(h,
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None, 
                             node_attr=h0)
            else:
                h, _ = layer(h, 
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None,
                             node_attr=None)

        h = self.node_dec(h)
        
        h = h*node_mask
        
        h = h.view(-1, n_nodes, self.hidden_nf) 

        h = torch.sum(h, dim=1)

        temp = self.graph_dec(h)

        pred = torch.zeros(n_reactions,  dtype=torch.float, device=self.device)

        pred = pred.index_add_(0, reaction_indexes, temp.squeeze(1)*reaction_indexes_signs)
        
        return pred, None
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.node_dec.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()
        for layer in self.graph_dec.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()

class EGNN_var2(EGNN_var1):
    def __init__(self, *args, **kwargs):
        super(EGNN_var2, self).__init__(*args, **kwargs)
        
    def forward(self, inputs):

        h0, positions, edges, n_nodes, node_mask, edge_mask, n_reactions, reaction_indexes, reaction_indexes_signs, _, _ = inputs
        
        h = self.embedding(h0) 

        for layer in self.layers:
            if self.node_attr:
                h, _ = layer(h, 
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None, 
                             node_attr=h0)
            else:
                h, _ = layer(h, 
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None,
                             node_attr=None)

        h = self.node_dec(h) 
        
        h = h*node_mask

        h = h.view(-1, n_nodes, self.hidden_nf)

        h = torch.sum(h, dim=1)  

        temp = torch.zeros(n_reactions, h.size(1), dtype=torch.float, device=self.device)

        temp = temp.index_add_(0, reaction_indexes, h * reaction_indexes_signs.unsqueeze(1))

        pred = self.graph_dec(temp)

        return pred.squeeze(1), None
    
class EGNN_atomic(EGNN_var1):
    def __init__(self, *args, **kwargs):
        super(EGNN_atomic, self).__init__(*args, **kwargs)
        
    def forward(self, inputs):

        h0, positions, edges, n_nodes, node_mask, edge_mask, n_reactions, reaction_indexes, reaction_indexes_signs, _, _ = inputs
        
        h = self.embedding(h0) 

        for layer in self.layers: 
            if self.node_attr:
                h, _ = layer(h,
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None, 
                             node_attr=h0)
            else:
                h, _ = layer(h, 
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None,
                             node_attr=None)

        h = self.node_dec(h) 

        h = h*node_mask

        ac = self.graph_dec(h) 

        ac = ac.view(-1, n_nodes) 

        temp = torch.sum(ac, dim=1) 

        pred = torch.zeros(n_reactions,  dtype=torch.float, device=self.device)
        pred = pred.index_add_(0, reaction_indexes, temp*reaction_indexes_signs)
        
        reaction_contributions = [[] for _ in range(n_reactions)]

        for i in range(ac.size(0)):
            reaction_index = reaction_indexes[i].item()
            reaction_contributions[reaction_index].append(ac[i, :])
        
        return pred, reaction_contributions  

class EGNN_group(EGNN_var1):
    def __init__(self, *args, **kwargs):
        super(EGNN_group, self).__init__(*args, **kwargs)
        
    def forward(self, inputs):

        h0, positions, edges, n_nodes, node_mask, edge_mask, n_reactions, reaction_indexes, reaction_indexes_signs, group_adj, group_mask = inputs
        
        h = self.embedding(h0) 

        for layer in self.layers:
            if self.node_attr:
                h, _ = layer(h,
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None, 
                             node_attr=h0)
            else:
                h, _ = layer(h, 
                             edges, 
                             positions, 
                             edge_mask,
                             edge_attr=None,
                             node_attr=None)

        h = self.node_dec(h) 
        h = h*node_mask 
        
        hc = unsorted_segment_sum(h[group_adj[0]], group_adj[1], num_segments=h.size(0))

        gc = self.graph_dec(hc)

        gc = gc*group_mask

        gc = gc.view(-1, n_nodes)

        temp = torch.sum(gc, dim=1)

        pred = torch.zeros(n_reactions,  dtype=torch.float, device=self.device)
        pred = pred.index_add_(0, reaction_indexes, temp*reaction_indexes_signs)

        reaction_contributions = [[] for _ in range(n_reactions)]

        for i in range(gc.size(0)):
            reaction_index = reaction_indexes[i].item()
            reaction_contributions[reaction_index].append(gc[i, :])
        
        return pred, reaction_contributions   

class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, 
                 input_nf, 
                 hidden_nf, 
                 edges_in_d=0, 
                 nodes_att_dim=0,
                 act_fn=nn.SiLU(), 
                 attention=False, 
                 norm_diff=False):

        super(E_GCL, self).__init__()

        self.input_nf       = input_nf 
        self.output_nf      = input_nf 
        self.hidden_nf      = hidden_nf 
        self.edges_in_d     = edges_in_d
        self.nodes_att_dim  = nodes_att_dim
        self.act_fn         = act_fn
        self.attention      = attention
        self.norm_diff      = norm_diff
        self.edge_coords_nf = 1 
        self.input_edge     = input_nf * 2 

        self.edge_mlp = nn.Sequential(
            nn.Linear(self.input_edge + self.edge_coords_nf + self.edges_in_d, self.hidden_nf),
            self.act_fn,
            nn.Linear(self.hidden_nf, self.hidden_nf),
            self.act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(self.hidden_nf + self.input_nf + self.nodes_att_dim, self.hidden_nf),
            self.act_fn,
            nn.Linear(self.hidden_nf, self.output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(self.hidden_nf, 1),
                nn.Sigmoid())
            
    def reset_parameters(self): 
        for layer in self.edge_mlp.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()
        for layer in self.node_mlp.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()
        if self.attention:
            for layer in self.att_mlp.children():
                if hasattr(layer, 'reset_parameters~'):
                    layer.reset_parameters()

    def edge_model(self, source, target, radial, edge_attr):

        if edge_attr is None:  
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        
        row, col = edge_index

        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)

        return out, agg

    def coord2radial(self, edge_index, coord):

        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_mask, edge_attr=None, node_attr=None):

        row, col = edge_index 
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, edge_attr    

def unsorted_segment_sum(data, segment_ids, num_segments, dim=0):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    shape = list(data.shape)
    shape[dim] = num_segments
    result = data.new_full(shape, 0)  # Init empty result tensor.
    result.index_add_(0, segment_ids, data)
    return result

def prepare_inputs_egnn(data, input_args):

    batch_size, n_nodes, _ = data['positions'].size()

    atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(input_args['device'], input_args['dtype'])

    one_hot = data['one_hot'].to(input_args['device'], input_args['dtype'])
    charges = data['charges'].to(input_args['device'], input_args['dtype'])
    h0 = preprocess_input(one_hot, charges, input_args['charge_power'], input_args['max_charge'], input_args['device'])
    h0 = h0.view(batch_size * n_nodes, -1).to(input_args['device'], input_args['dtype'])

    bond_edges = get_adj_matrix(n_nodes, batch_size)
    bond_edges = bond_edges.to(input_args['device'])

    atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(input_args['device'], input_args['dtype'])
    edge_mask = get_edge_mask(data['atom_mask'])
    edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1).to(input_args['device'], input_args['dtype'])

    group_mask = data['group_mask'].view(batch_size * n_nodes, -1).to(input_args['device'], input_args['dtype'])
    group_adj = data['group_adj'].to(input_args['device'])

    n_reactions = data['n_reactions'].to(input_args['device'])
    reaction_indexes = data['reaction_indexes'].to(input_args['device'])
    reaction_indexes_signs = data['reaction_indexes_signs'].to(input_args['device'])

    return [h0, 
            atom_positions, 
            bond_edges, 
            n_nodes, 
            atom_mask,
            edge_mask,
            n_reactions, 
            reaction_indexes, 
            reaction_indexes_signs, 
            group_adj, 
            group_mask]

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars

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

def get_egnn_model(args):
    kwargs = {
        "in_node_nf": 15,
        "in_edge_nf": 0,
        "hidden_nf": 128,
        "node_attr": 0,
        "device": 'cpu',
        "act_fn": nn.SiLU(),
        "n_layers": 7,
        "attention": True,
        "charge_power": 2
    }
    
    if 'charge_power' in args:
        kwargs['in_node_nf'] = (args['charge_power'] + 1) * args['num_species']

    for k, v in kwargs.items():
        args.setdefault(k, v)
    
    # Determine model type
    model_class = {"original-var1": EGNN_var1, 
                   "original-var2": EGNN_var2, 
                   "atomic": EGNN_atomic, 
                   "group": EGNN_group
                   }.get(args["model_type"])
    if not model_class:
        raise ValueError("Invalid model type specified.")
    
    model = model_class(**args)
    return model, args
