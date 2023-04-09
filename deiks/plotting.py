import torch
from torch import Tensor
import matplotlib.pyplot as plt

from deiks.data import DELTAS_2D, MASKING_CONSTANT_2D, sample_2d_vector


def unsqueeze_2d_sample(sample: Tensor) -> Tensor:
    """Given a 2D (sample) of shape (13,) maps it back to its original grid 
    form."""
    assert sample.shape == (13,)
    sample = sample[1:]
    unsqueezed = torch.full((5, 5), MASKING_CONSTANT_2D)
    for d, delta in enumerate(DELTAS_2D):
        unsqueezed[delta[0]+2, delta[1]+2] = sample[d]
    return unsqueezed

def visualize_2d_sample(distance_map: Tensor,
                        h: float,
                        i: int,
                        j: int,
                        ) -> None:
    axs = plt.subplots(1, 2, figsize=(10, 5))[1]
    input_vector = sample_2d_vector(distance_map, h, i, j)[0]
    axs[0].imshow(distance_map[i-2:i+3, j-2:j+3].detach().cpu().numpy())
    axs[0].axis('off')
    axs[1].imshow(unsqueeze_2d_sample(input_vector))
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()