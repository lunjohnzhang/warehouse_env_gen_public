from typing import Tuple, List

import gin
import torch
from torch import nn
from torchvision.models import resnet

from env_search.utils.network import int_preprocess
from env_search.utils import kiva_obj_types

@gin.configurable
class WarehouseConvolutional(nn.Module):
    """Model based on discriminator described in V. Volz, J. Schrum, J. Liu, S.
    M. Lucas, A. Smith, and S. Risi, “Evolving mario levels in the latent
    space of a deep convolutional generative adversarial network,” in
    Proceedings of the Genetic and Evolutionary Computation Conference, 2018.

    Args:
        i_size (int): size of input image
        nc (int): total number of objects in the environment
        ndf (int): number of output channels of initial conv2d layer
        n_extra_layers (int): number of extra layers with out_channels = ndf to
            add
        head_dimensions (List): List of dimensions of the objective and measure
            heads
    """
    def __init__(
        self,
        i_size: int = gin.REQUIRED,
        nc: int = gin.REQUIRED,
        ndf: int = gin.REQUIRED,
        n_extra_layers: int = gin.REQUIRED,
        head_dimensions: List = gin.REQUIRED,
    ):
        super().__init__()

        assert i_size % 16 == 0, "i_size has to be a multiple of 16"
        assert len(head_dimensions
                  ) > 1, "Size of head_dimensions list should at least be 2"

        self.i_size = i_size
        self.nc = nc
        self.model, feature_size = self._build_model(i_size, nc, ndf,
                                                     n_extra_layers)
        self.obj_head, self.measure_heads = self._build_heads(
            feature_size, head_dimensions)

    @staticmethod
    def _build_model(i_size, nc, ndf, n_extra_layers):
        model = nn.Sequential()
        # Input is nc x i_size x i_size
        model.add_module(
            f"initial:conv:{nc}-{ndf}",
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        )
        model.add_module(f"initial:relu:{ndf}", nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = i_size / 2, ndf

        # Add extra layers with out_channels = ndf
        for t in range(n_extra_layers):
            model.add_module(
                f"extra-layers-{t}:{cndf}:conv",
                nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False),
            )
            model.add_module(f"extra-layers-{t}:{cndf}:batchnorm",
                             nn.BatchNorm2d(cndf))
            model.add_module(
                f"extra-layers-{t}:{cndf}:relu",
                nn.LeakyReLU(0.2, inplace=True),
            )

        # Add more conv2d layers with exponentially more out_channels
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            model.add_module(
                f"pyramid:{in_feat}-{out_feat}:conv",
                nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False),
            )
            model.add_module(f"pyramid:{out_feat}:batchnorm",
                             nn.BatchNorm2d(out_feat))
            model.add_module(f"pyramid:{out_feat}:relu",
                             nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        model.add_module("output:flatten", nn.Flatten())
        feature_size = cndf * 4 * 4
        return model, feature_size

    @staticmethod
    def _build_heads(feature_size, head_dimensions):
        obj_head = nn.Linear(feature_size, head_dimensions[0])
        measure_heads = nn.ModuleList(
            [nn.Linear(feature_size, dim) for dim in head_dimensions[1:]])

        return obj_head, measure_heads

    def predict_objs_and_measures(
            self,
            lvls: torch.Tensor,
            aug_lvls: torch.Tensor = None) -> Tuple[torch.Tensor]:
        """Predicts objectives and measures when given int levels.

        Args:
            lvls: (n, lvl_height, lvl_width) tensor of int levels.
            aug_lvls: (n, nc_aug, lvl_height, lvl_width) tensor of predicted aug
                data. This data is concatenated with the onehot version of the
                level as additional channels to the network. Set to None to not
                use aug data. (default: None)
        Returns:
            predicted objectives and predicted measures
        """
        inputs = int_preprocess(lvls, self.i_size, self.nc,
                                kiva_obj_types.index("."))
        if aug_lvls is not None:
            inputs[:, -aug_lvls.shape[1]:, ...] = aug_lvls
        return self(inputs)

    def forward(self, inputs):
        """Runs the network on input images."""
        features = self.model(inputs)
        obj = self.obj_head(features)
        measures = tuple(e(features) for e in self.measure_heads)
        return obj, *measures


@gin.configurable
class WarehouseAugResnetOccupancy(nn.Module):
    """Resnet for predicting the agent cell occupancy (aka tile usage) on
    warehouse map.

    Args:
        i_size (int): size of input image.
        nc (int): number of input channels.
        ndf (int): number of output channels of conv2d layer.
        n_res_layers (int): number of residual layers (2x conv per residual
            layer).
        n_out (int): number of outputs.
    """

    def __init__(
        self,
        i_size: int = gin.REQUIRED,
        nc: int = gin.REQUIRED,
        ndf: int = gin.REQUIRED,
        n_res_layers: int = gin.REQUIRED,
        n_out: int = 1,
    ):
        super().__init__()

        assert i_size % 16 == 0, "i_size has to be a multiple of 16"

        self.i_size = i_size
        self.nc = nc
        self.model = self._build_model(nc, ndf, n_res_layers, n_out)

    @staticmethod
    def _build_model(nc, ndf, n_res_layers, n_out):
        model = nn.Sequential()
        # Input is nc x i_size x i_size
        model.add_module(
            f"initial:conv:{nc}-{ndf}",
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
        )
        model.add_module(f"initial:relu:{ndf}", nn.LeakyReLU(0.2, inplace=True))

        # Add extra layers with out_channels = ndf
        for t in range(n_res_layers):
            model.add_module(f"residual-layer-{t}", resnet.BasicBlock(ndf, ndf))

        model.add_module(
            f"final:1x1conv:{ndf}-{n_out}",
            nn.Conv2d(ndf, n_out, 1, 1, 0, bias=False),
        )

        return model

    def forward(self, inputs):
        """Runs the network on input images."""
        return self.model(inputs)

    def int_to_logits(self, lvls: torch.Tensor) -> torch.Tensor:
        _, lvl_height, lvl_width = lvls.shape
        outputs = self.int_to_no_crop(lvls)
        return outputs[:, :, :lvl_height, :lvl_width]

    def int_to_no_crop(self, lvls: torch.Tensor) -> torch.Tensor:
        inputs = int_preprocess(lvls, self.i_size, self.nc,
                                kiva_obj_types.index("."))
        return self(inputs)

    def load_from_saved_weights(self):
        return self


@gin.configurable
class WarehouseAugResnetRepairedMapAndOccupancy(nn.Module):
    """Resnet for predicting the agent cell occupancy (aka tile usage) on
    warehouse map.

    Args:
        i_size (int): size of input image.
        nc (int): number of input channels.
        ndf (int): number of output channels of conv2d layer.
        n_res_layers (int): number of residual layers (2x conv per residual
            layer).
        n_out (int): number of outputs.
    """

    def __init__(
        self,
        i_size: int = gin.REQUIRED,
        nc: int = gin.REQUIRED,
        ndf: int = gin.REQUIRED,
        n_res_layers: int = gin.REQUIRED,
        n_out: int = 1,
    ):
        super().__init__()

        assert i_size % 16 == 0, "i_size has to be a multiple of 16"

        self.i_size = i_size
        self.nc = nc
        self.repaired_map_pred_mdl = self._build_model(
            nc, ndf, n_res_layers, nc)
        self.occupancy_pred_mdl = self._build_model(
            nc, ndf, n_res_layers, n_out)

    @staticmethod
    def _build_model(nc, ndf, n_res_layers, n_out):
        model = nn.Sequential()
        # Input is nc x i_size x i_size
        model.add_module(
            f"initial:conv:{nc}-{ndf}",
            nn.Conv2d(nc, ndf, 3, 1, 1, bias=False),
        )
        model.add_module(f"initial:relu:{ndf}", nn.LeakyReLU(0.2, inplace=True))

        # Add extra layers with out_channels = ndf
        for t in range(n_res_layers):
            model.add_module(f"residual-layer-{t}", resnet.BasicBlock(ndf, ndf))

        model.add_module(
            f"final:1x1conv:{ndf}-{n_out}",
            nn.Conv2d(ndf, n_out, 1, 1, 0, bias=False),
        )

        return model

    def forward(self, inputs):
        """Runs the network on input images."""
        pred_repaired_map = self.repaired_map_pred_mdl(inputs)
        occupancy = self.occupancy_pred_mdl(pred_repaired_map)
        return pred_repaired_map, occupancy

    def int_to_logits(self, lvls: torch.Tensor) -> torch.Tensor:
        _, lvl_height, lvl_width = lvls.shape
        pred_repaired_map, occupancy = self.int_logits_to_no_crop(lvls)
        return (pred_repaired_map[:, :, :lvl_height, :lvl_width],
                occupancy[:, :, :lvl_height, :lvl_width])

    def int_to_map_no_crop(self, lvls: torch.Tensor) -> torch.Tensor:
        _, lvl_height, lvl_width = lvls.shape

        # Take the argmax on the repaired map to the get the actual repaired map
        pred_repaired_map, occupancy = self.int_logits_to_no_crop(lvls)
        pred_repaired_map = torch.argmax(pred_repaired_map, dim=1)
        return pred_repaired_map, occupancy

    def int_logits_to_no_crop(self, lvls: torch.Tensor) -> torch.Tensor:
        inputs = int_preprocess(lvls, self.i_size, self.nc,
                                kiva_obj_types.index("."))
        pred_repaired_map, occupancy = self(inputs)
        return pred_repaired_map, occupancy

    def load_from_saved_weights(self):
        return self