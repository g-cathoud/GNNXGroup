import torch
from torch import nn

"""
This model was extracted from:
https://github.com/vgsatorras/egnn
"""

class EGNN(nn.Module):
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

        super(EGNN, self).__init__()

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

        h0, positions, edges, n_nodes, node_mask, edge_mask, _, _ = inputs
        
        h = self.embedding(h0) # Take the initial embedding of each node and expand it to the hidden dimention

        for layer in self.layers: # Massage passing, where only the embedding of each node is updated
            if self.node_attr:
                h, _ = layer(h, # Current embeddings to be updates
                             edges, # Adjacency matrix/list
                             positions, # Coordinates of the atoms
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

        h = self.node_dec(h) # Last pure NN layer in each node embedding 
                             # before extracting the property of the graph
        h = h*node_mask # Extract information of only the nodes that matter
        h = h.view(-1, n_nodes, self.hidden_nf) # Correct dimentions after batch stacking

        h = torch.sum(h, dim=1) # This summs the embedings of all the nodes into just one. 
        pred = self.graph_dec(h) # Extract the property from the summ trough a NN
        pred = pred.squeeze(1)
        
        return pred, None
    
    def reset_parameters(self): # Linear parameters will be initilized with kaiming_uniform method
        self.embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for layer in self.node_dec.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()
        for layer in self.graph_dec.children():
            if hasattr(layer, 'reset_parameters~'):
                layer.reset_parameters()

class EGNN_atomic(EGNN):
    def __init__(self, *args, **kwargs):
        super(EGNN_atomic, self).__init__(*args, **kwargs)
        
    def forward(self, inputs):

        h0, positions, edges, n_nodes, node_mask, edge_mask, _, _ = inputs
        
        h = self.embedding(h0) # Take the initial embedding of each node and expand it to the hidden dimention

        for layer in self.layers: # Massage passing, where only the embedding of each node is updated
            if self.node_attr:
                h, _ = layer(h, # Current embeddings to be updates
                             edges, # Adjacency matrix/list
                             positions, # Coordinates of the atoms
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

        h = self.node_dec(h) # Last pure NN layer in each node embedding 
                             # before extracting the property of the graph
        h = h*node_mask # Extract information of only the nodes that matter
        ac = self.graph_dec(h) # Extract the atomic contributions

        ac = ac*node_mask # Extract information of only the nodes that matter

        ac = ac.view(-1, n_nodes) # Correct dimentions after batch stacking

        pred = torch.sum(ac, dim=1) # This sums the atomic contributions to get the final prediction
        
        return pred, ac

class EGNN_group(EGNN):
    def __init__(self, *args, **kwargs):
        super(EGNN_group, self).__init__(*args, **kwargs)
        
    def forward(self, inputs):

        h0, positions, edges, n_nodes, node_mask, edge_mask, group_adj, group_mask = inputs
        
        h = self.embedding(h0) # Take the initial embedding of each node and expand it to the hidden dimention

        for layer in self.layers: # Massage passing, where only the embedding of each node is updated
            if self.node_attr:
                h, _ = layer(h, # Current embeddings to be updates
                             edges, # Adjacency matrix/list
                             positions, # Coordinates of the atoms
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

        h = self.node_dec(h) # Last pure NN layer in each node embedding 
                             # before extracting the property of the graph
        h = h*node_mask # Extract information of only the nodes that matter
        
        hc = unsorted_segment_sum(h[group_adj[0]], group_adj[1], num_segments=h.size(0)) # Sum embeddings based on group information

        gc = self.graph_dec(hc) # Extract the grouo contributions

        gc = gc*group_mask # Extract information of only the nodes that matter

        gc = gc.view(-1, n_nodes) # Correct dimentions after batch stacking

        pred = torch.sum(gc, dim=1) # This sums the group contributions to get the final prediction 
        
        return pred, gc   

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
                 # output_nf, # Usually the same as input nf. 
                 # We don't want to change the embedings at this phase.
                 hidden_nf, 
                 edges_in_d=0, # We could add some attrbutes to the edge. 
                               # E.g. put the information of bonds types.
                 nodes_att_dim=0, # We could also use the original embeddings of 
                                  # the nodes on the trainings.
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
        self.edge_coords_nf = 1 # Usually the same as input nf.
        self.input_edge     = input_nf * 2 # The size is double to account for the embeddings of i and j

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
            
    def reset_parameters(self): # Linear parameters will be initilized with kaiming_uniform method
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
        # The edge model takes the embeddings of the source and target and nodes 
        # and also the "edge atributes" to create a new embedding for this edge.

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
        # Then, with the embeddings of the edges, the embeddigs of the nodes are updated based on 
        # the nrighboors edges of each node. 
        
        row, col = edge_index

        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        # Here I will summ all the edge atributes from the neighbors, since the graph in undirected, 
        # using col or row does not matter. x.size(0) means that I will summ the messages of the 
        # neighboors for all the atoms. I could also use the mean here: unsorted_segment_mean().

        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)

        return out, agg

    def coord2radial(self, edge_index, coord):
        # This function is used to get the distance between two nodes.
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_mask, edge_attr=None, node_attr=None):
        # As said, we start calculating embeddings for each edge. Then, based on these 
        # embeddings and the adj matrix, we update the embeding of the node.

        row, col = edge_index # row represents the source nodes, col represents the destination nodes 
        radial, coord_diff = self.coord2radial(edge_index, coord) # Gets the distances between the nodes

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

    return h0, atom_positions, bond_edges, n_nodes, atom_mask, edge_mask, group_adj, group_mask

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
    
    if args['dataset'] == 'qmugs':
        kwargs['in_node_nf'] = 27
    elif args['dataset'] == 'alchemy':
        kwargs['in_node_nf'] = 21

    for k, v in kwargs.items():
        args.setdefault(k, v)
    
    # Determine model type
    model_class = {"original": EGNN, "atomic": EGNN_atomic, "group": EGNN_group}.get(args["model_type"])
    if not model_class:
        raise ValueError("Invalid model type specified.")
    
    model = model_class(**args)
    return model, args
