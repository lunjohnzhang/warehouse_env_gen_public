"""Simple dataset for storing and sampling data for emulation model with aug
prediction."""
from collections import namedtuple

import numpy as np
import torch

from env_search.device import DEVICE
from env_search.warehouse.emulation_model.buffer import Buffer, Experience

AugExperience = namedtuple(
    "AugExperience", Experience._fields + ("occupancy",))

# Used for batches of items, e.g. a batch of levels, a batch of objectives.
# Solution is excluded since it is not used in training
BatchExperience = namedtuple("BatchExperience", AugExperience._fields[1:])


class AugBuffer(Buffer):
    """Stores data samples for training the emulation model including the aug
    predictor.

    Args:
        seed (int): Random seed to use (default None)
    """

    def __init__(
            self,
            seed: int = None,  # pylint: disable = unused-argument
    ):
        super().__init__(seed)
        self.occupancys = []

    def add(self, e: AugExperience):
        """Adds experience to the buffer."""
        super().add(e)
        self.occupancys.append(e.occupancy)

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
        )
