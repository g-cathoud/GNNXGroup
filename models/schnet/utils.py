import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_
    

class CosineCutoff(nn.Module):
    """ Just transforming the consine cutoff function into nn.module."""

    def __init__(self, cutoff):

        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))

    def cosine_cutoff(self, input, cutoff):
        """ Behler-style cosine cutoff. Math (latex):
            f(r) = \begin{cases}
                0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
                & r < r_\text{cutoff} \\
                0 & r \geqslant r_\text{cutoff} \\
                \end{cases}
            """

        # Compute values of cutoff function
        input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
        # Remove contributions beyond the cutoff radius
        input_cut *= (input < cutoff).float()
        return input_cut

    def forward(self, input):
        return self.cosine_cutoff(input, self.cutoff)
    

class GaussianRBF(nn.Module):
    r"""Gaussian radial basis functions."""

    def __init__(self, n_rbf, cutoff, start = 0.0, trainable = False):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`
            start: center of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def gaussian_rbf(self, inputs, offsets, widths):
        coeff = -0.5 / torch.pow(widths, 2)
        diff = inputs[..., None] - offsets
        y = torch.exp(coeff * torch.pow(diff, 2))
        return y

    def forward(self, inputs):
        return self.gaussian_rbf(inputs, self.offsets, self.widths)


class Dense(nn.Linear):
    r"""Fully connected linear layer with activation function. Math: y = activation(x W^T + b)"""

    def __init__(self,
        in_features,
        out_features,
        bias = True,
        activation = None,
        weight_init = xavier_uniform_,
        bias_init = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y

def build_mlp(
    n_in,
    n_out,
    n_hidden = None,
    n_layers = 2,
    activation = F.silu,
    last_bias = True,
    last_zero_init = False
    ):
    
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    layers = [
        Dense(n_neurons[i], n_neurons[i + 1], activation=activation)
        for i in range(n_layers - 1)
    ]
    # assign a Dense layer (without activation function) to the output layer

    if last_zero_init:
        layers.append(
            Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=last_bias,
            )
        )
    else:
        layers.append(
            Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net

def shifted_softplus(x):
    """
    Compute shifted soft-plus activation function. 
    Math: y = ln(1 + e^{-x}) - ln(2) 
    """
    return F.softplus(x) - math.log(2.0)

def softplus_inverse(x):
    """Inverse of the softplus function."""
    return x + (torch.log(-torch.expm1(-x)))

def unsorted_segment_sum(data, segment_ids, num_segments, dim=0):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    shape = list(data.shape)
    shape[dim] = num_segments
    result = data.new_full(shape, 0)  # Init empty result tensor.
    result.index_add_(0, segment_ids, data)
    return result