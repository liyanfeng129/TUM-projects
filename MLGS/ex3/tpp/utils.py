from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()


@typechecked
def get_sequence_batch(
    inter_event_times: List[TensorType[torch.float32]],
) -> Tuple[
    TensorType[torch.float32, "batch", "max_seq_length"],
    TensorType[torch.bool, "batch", "max_seq_length"],
]:
    """
    Generate padded batch and mask for list of sequences.

        Args:
            inter_event_times (List): list of inter-event times

        Returns:
            batch: batched inter-event times. shape [batch_size, max_seq_length]
            mask: boolean mask indicating inter-event times. shape [batch_size, max_seq_length]
    """

    #######################################################
    # write here
    batch = None
    mask = None # Pad sequences with zero padding
    batch = pad_sequence(inter_event_times, batch_first=True, padding_value=0.0)

    # Create mask indicating the positions of actual data (not padding)
    lengths = [seq.size(0) for seq in inter_event_times]
    max_seq_length = batch.size(1)
    mask = torch.zeros((len(inter_event_times), max_seq_length), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1


    #######################################################

    return batch, mask


@typechecked
def get_tau(
    t: TensorType[torch.float32, "sequence_length"], t_end: TensorType[torch.float32, 1]
) -> TensorType[torch.float32]:
    """
    Compute inter-eventtimes from arrival times

        Args:
            t: arrival times. shape [seq_length]
            t_end: end time of the temporal point process.

        Returns:
            tau: inter-eventtimes.
    """
    # compute inter-event times
    #######################################################
    # write here
    # if N = 2, t = (t1, t2), t_big = (t1, t2, t3), t_small = (t0, t1, t2), tau = t_big - t_small
    # Initialize t0 and t_end
    
    t0 = torch.tensor([0.0], device=t.device)
    # Ensure t_end is a tensor of shape [1]
    if t_end.dim() == 0:
        t_end = t_end.unsqueeze(0)

    # Create t_big and t_small for computing inter-event times
    t_big = torch.cat((t, t_end))
    t_small = torch.cat((t0, t))

    # Compute inter-event times
    tau = t_big - t_small

    return tau
