import torch
import tqdm
import matplotlib.pyplot as plt
from mlgs24ex3_ge74wan.tpp.utils import get_tau, get_sequence_batch
# load toy data
data = torch.load("data/hawkes.pkl")

arrival_times = data["arrival_times"]
t_end = data["t_end"]


get_tau_result = get_tau(arrival_times[0], t_end)
