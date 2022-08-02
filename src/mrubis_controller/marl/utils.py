from typing import Dict
import torch

from marl.replay_buffer import ReplayBuffer

def distance_rp_buffers(replay_buffers: Dict[str, ReplayBuffer]):
    buffers = sorted(replay_buffers.items(), key=lambda x: x[0])
    num_buffers = len(replay_buffers)
    distances = torch.zeros((num_buffers, num_buffers))
    distributions = [buffer[1].get_dist_probs()[0] for buffer in buffers]
    for index_tuple in distances.new_ones((num_buffers, num_buffers)).tril().nonzero():
        distances[index_tuple[0], index_tuple[1]] = torch.pow(distributions[index_tuple[0]] - distributions[index_tuple[1]], 2).sum().sqrt()
    return distances, list(replay_buffers.keys())
