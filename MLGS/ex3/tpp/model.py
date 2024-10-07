from typing import Tuple, Union

import torch
from torch.nn import functional as F
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
class NeuralTPP(torch.nn.Module):
    """Neural Temporal Point Process class
    Args:
        hidden_dim (int): Number of history_emb dimensions.
    """

    def __init__(self, hidden_dim: int = 16):
        super(NeuralTPP, self).__init__()

        self.hidden_dim = hidden_dim

        # Single layer RNN for history embedding with tanh nonlinearity
        #######################################################
        # write here and replace the default
        self.embedding_rnn = torch.nn.RNN(input_size=2, hidden_size=hidden_dim, batch_first=True)
        #######################################################

        # Single layer neural network to predict mu and log(sigma)
        #######################################################
        # write here and replace the default
        self.linear = torch.nn.Linear(hidden_dim, 2)
        #######################################################

        # value to be used for numerical problems
        self.eps = 1e-8

    def log_likelihood(
        self,
        times: TensorType[torch.float32, "batch", "max_seq_length"],
        mask: TensorType[torch.bool, "batch", "max_seq_length"],
    ) -> TensorType[torch.float32, "batch"]:
        """Compute the log-likelihood for a batch of padded sequences.
        Args:
            times (Tensor): Padded inter-event times,
                shape (batch_size, seq_len+1)
            mask (Tensor): Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len+1)
        Returns:
            log_likelihood: Log-likelihood for each sample in the batch,
                shape (batch_size,)
        """
        # clamp for stability
        times = torch.clamp(times, min=self.eps)

        # get history_emb
        history_emb = self.embed_history(times)

        # get cond. distributions
        mu, sigma = self.get_distribution_parameters(history_emb)
        dist = self.get_distributions(mu, sigma)

        # calculate negative log_likelihood
        log_density = self.get_log_density(dist, times, mask)
        log_survival = self.get_log_survival_prob(dist, times, mask)

        log_likelihood = log_density + log_survival

        return log_likelihood

    def get_log_density(
        self,
        distribution: torch.distributions.LogNormal,
        times: TensorType[torch.float32, "batch", "max_seq_length"],
        mask: TensorType[torch.bool, "batch", "max_seq_length"],
    ) -> TensorType["batch"]:
        """Compute the log-density for a batch of padded sequences.
        Args:
            distribution (torch.distributions.LogNormal): instance of pytorch distribution class
            times (Tensor): Padded inter-event times,
                shape (batch_size, seq_len+1)
            mask (Tensor): Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len+1)
            B (int): batch size
            seq_len (int): max sequence length
        Returns:
            log_density: Log-density for each sample in the batch,
                shape (batch_size,)
        """
        # calculate log density
        #######################################################
        # write here and replace the default
        log_density = None
         # Calculate log probability density
        log_prob = distribution.log_prob(times)

        # Mask out padding entries
        masked_log_prob = torch.where(mask, log_prob, torch.zeros_like(log_prob))

        # Set the last valid log density to zero for each row
        for i in range(times.size(0)):
            last_valid_index = mask[i].nonzero(as_tuple=True)[0][-1]
            masked_log_prob[i, last_valid_index] = 0

        # Sum log densities along the sequence length dimension
        log_density = masked_log_prob.sum(dim=-1)
        #######################################################
        return log_density

    def get_log_survival_prob(
        self,
        distribution: torch.distributions.LogNormal,
        times: TensorType[torch.float32, "batch", "max_seq_length"],
        mask: TensorType[torch.bool, "batch", "max_seq_length"],
    ) -> TensorType["batch"]:
        """Compute the log-intensities for a batch of padded sequences.
        Args:
            distribution (torch.distributions.LogNormal): instance of pytorch distribution class
            times (Tensor): Padded inter-event times,
                shape (batch_size, seq_len+1)
            mask (Tensor): Boolean mask that indicates which entries
                do NOT correspond to padding, shape (batch_size, seq_len+1)
            B (int): batch size
            seq_len (int): max sequence length
        Returns:
            log_surv_last: Log-survival probability for each sample in the batch,
                shape (batch_size,)
        """
        # calculate log survival probability
        #######################################################
        # write here and replace the default
        log_surv_last = None
         # Compute the CDF for each time step
        cdf_values = distribution.cdf(times)
        
        # Compute the survival probabilities
        survival_probs = 1 - cdf_values
        
        # Apply the mask
        valid_survival_probs = torch.where(mask, survival_probs, torch.ones_like(survival_probs))
        
        # Compute the log of the valid survival probabilities
        log_survival_probs = torch.log(valid_survival_probs)

        # Initialize a tensor to hold the last valid log-survival probability for each batch
        log_surv_last = torch.zeros(times.size(0), dtype=torch.float32)
        
        # Iterate over the batch to find the last valid log-survival probability
        for i in range(times.size(0)):
            last_valid_index = mask[i].nonzero(as_tuple=True)[0][-1]
            log_surv_last[i] = log_survival_probs[i, last_valid_index]
        
        #######################################################

        return log_surv_last

    def encode(
        self, times: TensorType[torch.float32, "batch", "max_seq_length"]
    ) -> TensorType[torch.float32, "batch", "max_seq_length", 2]:
        #######################################################
        # write here and replace the default
        # Add a small epsilon value to avoid taking logarithm of zero
        epsilon = self.eps
        times = times + epsilon
        log_times = torch.log(times)
        # Concatenate tau_i and log(tau_i) along the last dimension
        x = torch.stack((times, log_times), dim=-1)
        #######################################################
        return x

    def embed_history(
        self, times: TensorType[torch.float32, "batch", "max_seq_length"]
    ) -> TensorType[torch.float32, "batch", "max_seq_length", "history_emb_dim"]:
        """Embed history for a batch of padded sequences.
        Args:
            times: Padded inter-event times,
                shape (batch_size, max_seq_length)
        Returns:
            history_emb: history_emb embedding of the history,
                shape (batch_size, max_seq_length, embedding_dim)
        """

        #######################################################
        # write here and replace the default
        # Encode inter-event times
        encoded_times = self.encode(times)
        
        # Pass encoded times through RNN
        history_emb, _ = self.embedding_rnn(encoded_times)

        # Insert zero context c_1 at the beginning and remove c_N+1 from the end
        history_emb = torch.cat([torch.zeros_like(history_emb[:, :1]), history_emb[:, :-1]], dim=1)
        #######################################################

        return history_emb

    def get_distributions(
        self,
        mu: TensorType[torch.float32, "batch", "max_seq_length"],
        sigma: TensorType[torch.float32, "batch", "max_seq_length"],
    ) -> Union[torch.distributions.LogNormal, None]:
        """Get log normal distribution given mu and sigma.
        Args:
            mu (tensor): predicted mu (batch, max_seq_length)
            sigma (tensor): predicted sigma (batch, max_seq_length)

        Returns:
            Distribution: log_normal
        """

        #######################################################
        # write here and replace the default
        
        log_norm_dist = None
        log_norm_dist = torch.distributions.LogNormal(mu, sigma)
        #######################################################
        return log_norm_dist

    def get_distribution_parameters(
        self,
        history_emb: TensorType[
            torch.float32, "batch", "max_seq_length", "history_emb_dim"
        ],
    ) -> Tuple[
        TensorType[torch.float32, "batch", "max_seq_length"],
        TensorType[torch.float32, "batch", "max_seq_length"],
    ]:
        """Compute distribution parameters.
        Args:
            history_emb (Tensor): history_emb tensor,
                shape (batch_size, seq_len+1, C)
        Returns:
            Parameter (Tuple): mu, sigma
        """
        #######################################################
        # write here and replace the default
        mu = None
        sigma = None
         # Apply linear layer to history embeddings to obtain mu and log(sigma)
        output = self.linear(history_emb)
    
        # Split output into mu and log(sigma)
        mu, log_sigma = torch.split(output, 1, dim=-1)
    
        # Compute sigma by exponentiating log(sigma)
        sigma = torch.exp(log_sigma)

        mu = mu.squeeze(-1)
        sigma = sigma.squeeze(-1)
        #######################################################
        return mu, sigma

    def forward(self):
        """
        Not implemented
        """
        pass
