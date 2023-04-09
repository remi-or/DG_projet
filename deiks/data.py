from typing import Tuple, Callable
import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from random import randint


#region 2D functions

DELTAS_2D = [(-2, 0), 
             (-1, -1), (-1, 0), (-1, 1), 
             (0, -2), (0, -1), (0, 1), (0, 2), 
             (1, -1), (1, 0), (1, 1), 
             (2, 0)]
MASKING_CONSTANT_2D = -0.7

def regular_grid_(h: float) -> Tensor:
    """Returns a regular grid spanning in [0,1]^2 parametrized by (h)."""
    return torch.stack(torch.meshgrid(torch.arange(0, 1, h),
                                      torch.arange(0, 1, h),
                                      indexing='ij')).transpose(0, 2)

def source_set_fn(h: float,
                 function: Callable[[Tensor], Tensor],
                 ) -> Tensor:
    """Returns a tensor [xs, ys] where xs is a regular subdivison of [0, 1] 
    parametrized by (h) and ys is (function) applied to xs and normalized."""
    xs = torch.arange(0, 1, h)
    ys = function(xs)
    ys = (ys - ys.min()) / (ys.max() - ys.min())
    return torch.vstack((xs[None], ys[None])).T

def distance_map_set(h: float,
                     source_set: Tensor,
                     ) -> Tensor:
    """Given a set of source points (source_set) and a parameter (h), returns
    the distance map of a grid in [0,1]^2 parametrized by (h) to the source
    set."""
    grid = regular_grid_(h)
    distance_map = torch.full_like(grid[:,:,0], torch.inf)
    for s in source_set:
        dmap = (grid - s).square().sum(dim=-1)
        distance_map = distance_map.minimum(dmap)
    return distance_map.sqrt()

def distance_map_fn(h: float,
                    source_function: Callable[[Tensor], Tensor],
                    source_refinement: float = 1.0,
                    ) -> Tensor:
    """Pipelines [source_set_fn] and [distance_map_set]."""
    source_set = source_set_fn(h / source_refinement, source_function)
    return distance_map_set(h, source_set)

def compute_distance_maps(h: float) -> Tensor: # S, G, G
    """Outputs 10 different distance maps parametrized by (h)."""
    distances_maps = []
    # Analytic sources
    source_functions = [torch.square, 
                        torch.sqrt,
                        lambda xs : torch.cos(torch.pi * xs),
                        lambda xs : torch.sin(torch.pi * xs),
                        lambda xs : torch.tan(0.47 * torch.pi * xs)]
    for fn in source_functions:
        distances_maps.append(distance_map_fn(h, fn))
    # Random sources
    for i in torch.randint(1, 20, (len(source_functions),)):
        source_set = torch.rand((i.item(), 2))
        distances_maps.append(distance_map_set(h, source_set))
    return torch.stack(distances_maps)

def normalize_distances_2d(distances: Tensor, # D,
                           masking_distance: float, # 1
                           ) -> Tuple[Tensor, float, float]:
    """Normalizes (distances) according DES. The distance over which to mask 
    with MASKING_CONSTANT_2D is (masking_distance)"""
    mask = distances > masking_distance
    subtrahend = distances.min().item()
    distances = distances - subtrahend
    divisor = distances.mean() * 2
    if divisor == 0:
        divisor = 1e-8
    distances = distances / divisor
    distances = distances.masked_fill(mask, MASKING_CONSTANT_2D)
    return distances, subtrahend, divisor

def sample_2d_vector(distance_map: Tensor,
                     h: float,
                     i: int,
                     j: int,
                     ) -> Tuple[Tensor, float, float]:
    """Given a (distance_map) resting on a grid parametrized by (h) and
    coordinnates (i,j), samples an input vector for the 2D Deep Eikonal 
    Solver."""
    y = distance_map[i, j].item()
    input_vector = torch.zeros(13)
    input_vector[0] = h
    for d, (di, dj) in enumerate(DELTAS_2D):
        try:
            input_vector[d+1] = distance_map[i+di, j+dj]
        except IndexError:
            # print('idexing error...')
            input_vector[d+1] = MASKING_CONSTANT_2D
    # print(f"input vector h before normalization: {input_vector[0].item()}")
    input_vector[1:], subtrahend, divisor = normalize_distances_2d(input_vector[1:], y)
    # print(f"input vector h after normalization: {input_vector[0].item()}")
    return input_vector, subtrahend, divisor

def sample_2d_batch(distance_map: Tensor,
                    h: float,
                    batch_size: int, 
                    ) -> Tensor: # batch_size, 13
    """Given a (distance_map) resting on a grid parametrized by (h) and a 
    (batch_size) samples a batch of input vectors for the 2D Deep Eikonal 
    Solver."""
    G = distance_map.size(0)
    # accumulators
    batch = torch.zeros((batch_size, 13))
    gts = torch.zeros((batch_size, 1))
    # vector loop
    for b in range(batch_size):
        i = randint(2, G-3)
        j = randint(2, G-3) 
        input_vector, subtrahend, divisor = sample_2d_vector(distance_map, h, i, j)
        batch[b] = input_vector
        gts[b] = (distance_map[i, j].item() - subtrahend) / divisor
    return batch.to(distance_map.device), gts.to(distance_map.device)

#endregion

#region 3D functions

def generate_batch(data: Data,
                   batch_size: int,
                   cst: float = -0.25,
                   device: str = 'cpu',
                   ) -> Tuple[Tensor, Tensor]:
    vert, edge, dist = data.pos, data.edge_index, data.x
    xs, ys, hmax = [], [], -1
    for _ in range(batch_size):
        # choose center node
        i = randint(1, vert.size(0))-1
        # form subgraph
        nodes = k_hop_subgraph(i, 2, edge)[0]
        neighborhood = torch.hstack((vert[nodes], dist[nodes]))
        neighborhood, subtrahend, divisor = normalize_neighborhood(vert[i], neighborhood)
        y = (dist[i] - subtrahend) / divisor
        neighborhood[..., 3] = neighborhood[..., 3].masked_fill(neighborhood[..., 3] > y, cst)
        # add sample to batch
        xs.append(neighborhood)
        ys.append(y)
        # keep track of max dim
        hmax = max(hmax, nodes.size(0))
    # collate batch
    x_batch = torch.zeros((batch_size, hmax, 4))
    x_batch[:, :, 3].fill_(cst)
    for i, x in enumerate(xs):
        x_batch[i, :x.size(0)] = x
    return x_batch.to(device), torch.tensor(ys, device=device).reshape((-1, 1))

def normalize_distances(distances: Tensor, # D,
                        ) -> Tuple[Tensor, float, float]:
    subtrahend = distances.min().item()
    distances = distances - subtrahend
    divisor = distances.mean() * 2
    distances = distances / divisor
    return distances, subtrahend, divisor

def normalize_neighborhood(node_pos: Tensor,
                           neighborhood: Tensor,
                           ) -> Tuple[Tensor, float, float]:
    # normalize vertices
    neighborhood[..., :3] = neighborhood[..., :3] - node_pos
    # normalize distances
    neighborhood[..., 3], subtrahend, divisor = normalize_distances(neighborhood[..., 3])
    # find and remove self-referential node
    mask = (neighborhood[..., :3] != torch.zeros(3, device=neighborhood.device)).all(dim=-1)
    neighborhood = neighborhood[mask]
    return neighborhood, subtrahend, divisor

#endregion