import torch
from torch import nn
from torch import sparse as sp
from torch.nn import functional as F
from collections import OrderedDict
from itertools import chain
from typing import List, Tuple
import numpy as np


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer: as proposed in [Kipf et al. 2017](https://arxiv.org/abs/1609.02907).

    Parameters
    ----------
    in_channels: int
        Dimensionality of input channels/features.
    out_channels: int
        Dimensionality of output channels/features.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, arguments: Tuple[torch.tensor, torch.sparse.FloatTensor]) -> torch.tensor:
        """
        Forward method.

        Parameters
        ----------
        arguments: Tuple[torch.tensor, torch.sparse.FloatTensor]
            Tuple of feature matrix `X` and normalized adjacency matrix `A_hat`

        Returns
        ---------
        X: torch.tensor
            The result of the message passing step
        """
        X, A_hat = arguments
        ##########################################################
        # YOUR CODE HERE
        X = self._linear(X)
        # Perform the matrix multiplication between the adjacency matrix and the transformed features
        X = torch.sparse.mm(A_hat, X)
        
        ##########################################################
        return X


class GCN(nn.Module):
    """
    Graph Convolution Network: as proposed in [Kipf et al. 2017](https://arxiv.org/abs/1609.02907).

    Parameters
    ----------
    n_features: int
        Dimensionality of input features.
    n_classes: int
        Number of classes for the semi-supervised node classification.
    hidden_dimensions: List[int]
        Internal number of features. `len(hidden_dimensions)` defines the number of hidden representations.
    activation: nn.Module
        The activation for each layer but the last.
    dropout: float
        The dropout probability.
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_dimensions: List[int] = [64],
                 activation: nn.Module = nn.ReLU(),
                 dropout: float = 0.5):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dimensions = hidden_dimensions
        self.dropout = dropout
        self.propagate = nn.ModuleList(
            # Input and hidden layers
            [
                nn.Sequential(OrderedDict([
                    (f'gcn_{idx}', GraphConvolution(in_channels=in_channels,
                                                    out_channels=out_channels)),
                    (f'activation_{idx}', activation),
                    (f'dropout_{idx}', nn.Dropout(p=dropout))
                ]))
                for idx, (in_channels, out_channels)
                in enumerate(zip([n_features] + hidden_dimensions[:-1], hidden_dimensions))
            ]
            # Output and hidden layer
            + [
                nn.Sequential(OrderedDict([
                    (f'gcn_{len(hidden_dimensions)}', GraphConvolution(in_channels=hidden_dimensions[-1],
                                                                       out_channels=n_classes))
                ]))
            ]
        )

    def _normalize(self, A: torch.sparse.FloatTensor) -> torch.tensor:
        """
        For calculating $\hat{A} = D^{-\frac{1}{2}} A D^{-\frac{1}{2}}$.

        Parameters
        ----------
        A: torch.sparse.FloatTensor
            Sparse adjacency matrix with added self-loops.

        Returns
        -------
        A_hat: torch.sparse.FloatTensor
            Normalized message passing matrix
        """
        ##########################################################
        # YOUR CODE HERE
        A_hat = 0
        # Calculate the degree vector D
        degrees = torch.sparse.sum(A, dim=1).to_dense()  # Sum along rows and convert to dense tensor
        degrees_sqrt = torch.sqrt(degrees).clamp(min=1e-12)  # Avoid division by zero and take square root
    
        # Create diagonal matrix D^{-0.5} as a dense tensor
        D_inv_sqrt = torch.diag(1.0 / degrees_sqrt)
    
        # Convert D^{-0.5} to a sparse tensor
        D_inv_sqrt_sparse = D_inv_sqrt.to_sparse()
    
        # Normalize adjacency matrix: A_hat = D^{-0.5} A D^{-0.5}
        A_hat = torch.sparse.mm(D_inv_sqrt_sparse, A)
        A_hat = torch.sparse.mm(A_hat, D_inv_sqrt_sparse)

        ##########################################################
        return A_hat

    def forward(self, X: torch.Tensor, A: torch.sparse.FloatTensor) -> torch.tensor:
        """
        Forward method.

        Parameters
        ----------
        X: torch.tensor
            Feature matrix `X`
        A: torch.sparse.FloatTensor
            adjacency matrix `A` (with self-loops)

        Returns
        ---------
        X: torch.tensor
            The result of the last message passing step (i.e. the logits)
        """
        ##########################################################
        # YOUR CODE HERE
        A_hat = self._normalize(A)
        for layer in self.propagate:
            X = layer((X, A_hat))
        
        ##########################################################
        return X


def train(model: nn.Module,
          X: torch.Tensor,
          A: torch.sparse.FloatTensor,
          labels: torch.Tensor,
          idx_train: np.ndarray,
          idx_val: np.ndarray,
          lr: float = 1e-3,
          weight_decay: float = 5e-4,
          patience: int = 50,
          max_epochs: int = 300,
          display_step: int = 10):
    """
    Train a model using standard training.

    Parameters
    ----------
    model: nn.Module
        Model which we want to train.
    X: torch.Tensor [n, d]
        Dense attribute matrix.
    A: torch.sparse.FloatTensor [n, n]
        Sparse adjacency matrix.
    labels: torch.Tensor [n]
        Ground-truth labels of all nodes,
    idx_train: np.ndarray [?]
        Indices of the training nodes.
    idx_val: np.ndarray [?]
        Indices of the validation nodes.
    lr: float
        Learning rate.
    weight_decay : float
        Weight decay.
    patience: int
        The number of epochs to wait for the validation loss to improve before stopping early.
    max_epochs: int
        Maximum number of epochs for training.
    display_step : int
        How often to print information.
    seed: int
        Seed

    Returns
    -------
    trace_train: list
        A list of values of the train loss during training.
    trace_val: list
        A list of values of the validation loss during training.
    """
    trace_train = []
    trace_val = []
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = np.inf
    for it in range(max_epochs):
        ##########################################################
        # YOUR CODE HERE
        loss_train = torch.tensor(0)
        loss_val = torch.tensor(0)
          
        # Compute training loss
        logits = model(X, A)  # Get the logits directly
        
        probs = F.softmax(logits, dim=-1)
        # Compute training loss using the probabilities
        # loss_train = F.cross_entropy(probs[idx_train], labels[idx_train])
        n_class = probs.shape[-1]
        log_probs = torch.log(probs)
        one_hot = torch.eye(n_class)[labels[idx_train]]

        loss_train = -torch.sum(one_hot*log_probs[idx_train])
        loss_train.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits_val = model(X, A)  # Get the logits for validation
            probs_val = F.softmax(logits_val, dim=-1)
            log_probs_val = torch.log(probs_val)
            one_hot_val = torch.eye(n_class)[labels[idx_val]]
            loss_val = - torch.sum(one_hot_val*log_probs_val[idx_val])
        ##########################################################

        trace_train.append(loss_train.detach().item())
        trace_val.append(loss_val.detach().item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_epoch = it
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}
        else:
            if it >= best_epoch + patience:
                break

        if display_step > 0 and it % display_step == 0:
            print(
                f'Epoch {it:4}: loss_train: {loss_train.item():.5f}, loss_val: {loss_val.item():.5f} ')

    # restore the best validation state
    model.load_state_dict(best_state)
    return trace_train, trace_val


def sparse_dropout(A: torch.sparse.FloatTensor, p: float, training: bool) -> torch.sparse.FloatTensor:
    drop_val = F.dropout(A._values(), p, training)
    A = torch.sparse_coo_tensor(A._indices(), drop_val, A.shape)
    return A


class PowerIterationPageRank(nn.Module):
    """
    Power itertaion module for propagating the labels.

    Parameters
    ----------
    dropout: float
        The dropout probability.
    alpha: float
        The teleport probability.
    n_propagation: int
        The number of iterations for approximating the personalized page rank.
    """

    def __init__(self,
                 dropout: float = 0.5,
                 alpha: float = 0.15,
                 n_propagation: int = 5):
        super().__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.n_propagation = n_propagation

    def forward(self, logits: torch.Tensor, A_hat: torch.sparse.FloatTensor) -> torch.tensor:
        """
        Forward method.

        Parameters
        ----------
        logits: torch.tensor
            The local logits (for each node).
        A_hat: torch.tensor
            The normalized adjacency matrix `A_hat`.

        Returns
        ---------
        logits: torch.tensor
            The propagated/smoothed logits.
        """
        ##########################################################
        # YOUR CODE HERE
        initial_logits = logits.clone()
        for _ in range(self.n_propagation):
            dropout_mask = torch.rand(A_hat.to_dense().shape) > self.dropout
            temp = A_hat.to_dense() * dropout_mask
            temp = temp.to_sparse()
            # Perform graph propagation
            assert False
            logits = torch.sparse.mm(A_hat, logits)
            
            # Combine with teleport probability alpha
            logits = self.alpha * initial_logits + (1 - self.alpha) * logits
        ##########################################################
        return logits


class APPNP(GCN):
    """
    Approximate Personalized Propagation of Neural Predictions: as proposed in [Klicpera et al. 2019](https://arxiv.org/abs/1810.05997).

    Parameters
    ----------
    n_features: int
        Dimensionality of input features.
    n_classes: int
        Number of classes for the semi-supervised node classification.
    hidden_dimensions: List[int]
        Internal number of features. `len(hidden_dimensions)` defines the number of hidden representations.
    activation: nn.Module
        The activation for each layer but the last.
    dropout: float
        The dropout probability.
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 hidden_dimensions: List[int] = [64],
                 activation: nn.Module = nn.ReLU(),
                 dropout: float = 0.5,
                 alpha: float = 0.1,
                 n_propagation: int = 5):
        super().__init__(n_features, n_classes)
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_dimensions = hidden_dimensions
        self.dropout = dropout
        self.transform_features = (
            # Input dropout
            nn.Sequential(OrderedDict([
                (f'dropout_{0}', nn.Dropout(p=self.dropout))
            ]
                # Hidden layers
                + list(chain(*[
                    [(f'linear_{idx}', nn.Linear(in_features=in_features, out_features=out_features)),
                     (f'activation_{idx}', activation)]
                    for idx, (in_features, out_features)
                    in enumerate(zip([n_features] + hidden_dimensions[:-1], hidden_dimensions))
                ]))
                # Last layer
                + [
                (f'linear_{len(hidden_dimensions)}', nn.Linear(in_features=hidden_dimensions[-1],
                                                               out_features=n_classes)),
                (f'dropout_{len(hidden_dimensions)}',
                 nn.Dropout(p=self.dropout)),
            ]))
        )
        self.propagate = PowerIterationPageRank(dropout=dropout,
                                                alpha=alpha,
                                                n_propagation=n_propagation)

    def forward(self, X: torch.Tensor, A: torch.sparse.FloatTensor) -> torch.tensor:
        """
        Forward method.

        Parameters
        ----------
        X: torch.tensor
            Feature matrix `X`
        A: torch.tensor
            adjacency matrix `A` (with self-loops)

        Returns
        ---------
        logits: torch.tensor
            The propagated logits.
        """
        ##########################################################
        # YOUR CODE HERE
        logits = torch.tensor(0.0)
         # Transform the features
        X = self.transform_features(X)
        
        # Normalize the adjacency matrix
        A_hat = self._normalize(A)
        
        # Propagate the transformed features
        logits = self.propagate(X, A_hat)
        ##########################################################
        return logits


def calc_accuracy(logits: torch.Tensor, labels: torch.Tensor, idx_test: np.ndarray) -> float:
    """
    Calculates the accuracy.

    Parameters
    ----------
    logits: torch.tensor
        The predicted logits.
    labels: torch.tensor
        The labels vector.
    idx_test: torch.tensor
        The indices of the test nodes.
    """
    ##########################################################
    # YOUR CODE HERE
    accuracy = 0
    # Select logits and labels for the test nodes
    logits_test = logits[idx_test]
    labels_test = labels[idx_test]
    
    # Predicted labels are the argmax of logits along the class dimension
    predicted_labels = logits_test.argmax(dim=1)
    
    # Calculate accuracy
    correct = (predicted_labels == labels_test).sum().item()
    total = len(idx_test)
    accuracy = correct / total
    ##########################################################
    return accuracy
