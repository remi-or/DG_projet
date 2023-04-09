import torch
from torch import nn, Tensor

from deiks.data import sample_2d_batch


def eval_2D(model: nn.Module,
            distance_maps: Tensor,
            h: float,
            eval_samples: int,
            ) -> float:
    model.eval()
    criterion = nn.MSELoss()
    eval_loss = 0
    with torch.no_grad():
        eval_batch_size = eval_samples // distance_maps.shape[0]
        for distance_map in distance_maps:
            x, y = sample_2d_batch(distance_map, h, eval_batch_size)
            pred = model(x)
            eval_loss += criterion(pred, y).item()
    eval_loss /= distance_maps.size(0)
    return eval_loss

def train_2D(model: nn.Module,
             optimizer: torch.optim.Optimizer,
             distance_maps: Tensor, # S, G, G
             h: float,
             cycles: int,
             batch_per_cycle: int,
             batch_size: int,
             ):
    cycle_losses = []
    criterion = nn.MSELoss()
    # cycles loop
    for cycle in range(cycles):
        cycle_losses.append([])

        # train
        model.train()
        # distance_map loop
        for distance_map in distance_maps:
            distance_map_loss = 0
            # batches loop
            for batch in range(batch_per_cycle):
                optimizer.zero_grad()
                x, y = sample_2d_batch(distance_map, h, batch_size)
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                distance_map_loss += loss.item()
            cycle_losses[-1].append(distance_map_loss /batch_size)

        # display and eval
        avg_cycle_loss = sum(cycle_losses[-1]) / distance_maps.size(0)
        eval_loss = eval_2D(model, distance_maps, h, 1000)
        print(f"cycle {cycle} - loss {avg_cycle_loss:.4f} - eval_loss {eval_loss:.4f}")