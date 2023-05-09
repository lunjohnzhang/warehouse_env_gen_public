"""Utils for networks."""
import torch


def int_preprocess(int_lvls: torch.Tensor, i_size: int,
                   nc: int, padding: int) -> torch.Tensor:
    """
    Preprocess int levels for networks.
    Args:
        int_lvls: Input int levels (batch_size, lvl_height, lvl_width)
        i_size: Image size used by the network
        nc: Number of objects
        padding: Int value of the object to use as padding

    Returns:
        One-hot encoded and padded levels
    """
    _, lvl_height, lvl_width = int_lvls.shape
    onehot = torch.eye(nc, device=int_lvls.device)[
        int_lvls.long()]  # (n, lvl_height, lvl_width, nc)
    onehot = torch.moveaxis(onehot, 3, 1)  # (n, nc, lvl_height, lvl_width)

    inputs = torch.zeros((len(int_lvls), nc, i_size, i_size),
                         device=int_lvls.device)
    # Pad the levels with empty tiles.
    inputs[:, padding, :, :] = 1.0
    inputs[:, :, :lvl_height, :lvl_width] = onehot
    return inputs

def freeze_params(network):
    for param in network.parameters():
        param.requires_grad = False


def unfreeze_params(network):
    for param in network.parameters():
        param.requires_grad = True
