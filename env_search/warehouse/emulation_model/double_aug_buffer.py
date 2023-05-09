"""Simple dataset for storing and sampling data for emulation model with aug
prediction."""
from collections import namedtuple

import numpy as np
import torch

from env_search.device import DEVICE
from env_search.warehouse.emulation_model.aug_buffer import AugBuffer, AugExperience

DoubleAugExperience = namedtuple(
    "AugExperience", AugExperience._fields + ("repaired_map",))

# Used for batches of items, e.g. a batch of levels, a batch of objectives.
# Solution is excluded since it is not used in training
BatchExperience = namedtuple("BatchExperience", DoubleAugExperience._fields[1:])


class DoubleAugBuffer(AugBuffer):
    """Stores data samples for training the emulation model including the two aug predictors for repaired map and repaired_map.

    Args:
        seed (int): Random seed to use (default None)
    """

    def __init__(
            self,
            seed: int = None,  # pylint: disable = unused-argument
    ):
        super().__init__(seed)
        self.repaired_maps = []

    def add(self, e: DoubleAugExperience):
        """Adds experience to the buffer."""
        super().add(e)
        self.repaired_maps.append(e.repaired_map)

    def to_tensors(self):
        """Converts all buffer data to tensors."""
        # Convert to np.array due to this warning: Creating a tensor from a list
        # of numpy.ndarrays is extremely slow. Please consider converting the
        # list to a single numpy.ndarray with numpy.array() before converting to
        # a tensor.
        return BatchExperience(
            torch.as_tensor(np.array(self.levels),
                            device=DEVICE,
                            dtype=torch.float),
            torch.as_tensor(np.array(self.objectives),
                            device=DEVICE,
                            dtype=torch.float),
            torch.as_tensor(np.array(self.measures),
                            device=DEVICE,
                            dtype=torch.float),
            torch.as_tensor(np.array(self.occupancys),
                            device=DEVICE,
                            dtype=torch.float),
            torch.as_tensor(np.array(self.repaired_maps),
                            device=DEVICE,
                            dtype=torch.float),
        )
