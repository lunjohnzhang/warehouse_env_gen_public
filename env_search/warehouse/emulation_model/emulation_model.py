"""Provides primary WarehouseEmulationModel."""
import logging
import pickle as pkl
from pathlib import Path
from typing import List, Union

import cloudpickle
import gin
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from env_search.device import DEVICE
from env_search.warehouse.emulation_model.aug_buffer import AugBuffer
from env_search.warehouse.emulation_model.double_aug_buffer import DoubleAugBuffer
from env_search.warehouse.emulation_model.buffer import Buffer, Experience
from env_search.warehouse.emulation_model.networks import (
    WarehouseConvolutional, WarehouseAugResnetOccupancy,
    WarehouseAugResnetRepairedMapAndOccupancy)

from env_search.utils.network import (int_preprocess, freeze_params,
                                      unfreeze_params)
from env_search.utils import kiva_obj_types

logger = logging.getLogger(__name__)
# Just to get rid of pylint warning about unused import
NETWORKS = (WarehouseConvolutional, )


@gin.configurable(denylist=["seed"])
class WarehouseEmulationModel:
    """Class for warehouse emulation model.

    Args:
        network_type (type): Network class for the model. Intended
            for Gin configuration. May also be a callable which takes in no
            parameters and returns a new network.
        prediction_type (str): Type of prediction to use:
            "regression" to interpret network outputs as objective and measures
                and use MSE loss
            "classification" to interpret network outputs as logits
                corresponding to the objective and measure classes and use
                cross entropy loss
        train_epochs (int): Number of times to iterate over the dataset in
            train().
        train_batch_size (int): Batch size for each epoch of training.
        train_sample_size (int): Number of samples to choose for training. Set
            to None if all available data should be used. (default: None)
        train_sample_type (str): One of the following
            "random": Choose `n` uniformly random samples;
            "recent": Choose `n` most recent samples;
            "weighted": Choose samples s.t., on average, all samples are seen
                the same number of times across training iterations.
            (default: "recent")
        archive_dims (list): Number of cells in each dimension of the archive.
            Assumes the sizes of measure heads is the same. Only applicable if
            prediction_type is "classification".
        archive_ranges (list): Ranges of the archive. Used to convert the cell
            prediction to values. Only applicable if prediction_type is
            "classification".
        pre_network_type: Network class to use for pre-processing the inputs.
            Can be used for agent path prediction, etc. Input
            is unchanged if no class is passed. (default: None)
        pre_network_loss_func: One of "ce", "mse", "KL" or a callable.
            "ce": Cross entropy loss.
            "mse": MSE loss with target as cell visit frequency instead of
                probabilities.
            "KL": calculate the KL divergence loss of input and target
            callable: Any class similar to a pytorch loss. It will be
                initialized as `loss = pre_network_loss_func(reduction)` and
                then called during each training iteration as `loss(predictions,
                target)`.
        pre_repair_network_loss_func: One of "ce", "mse", "KL" or a callable.
            "ce": Cross entropy loss.
            "mse": MSE loss with target as cell visit frequency instead of
                probabilities.
            "KL": calculate the KL divergence loss of input and target
            callable: Any class similar to a pytorch loss. It will be
                initialized as `loss = pre_repair_network_loss_func(reduction)`
                and then called during each training iteration as
                `loss(predictions, target)`, where `predictions` and `target`
                are assumed to be predicted and true repaired maps.
        pre_network_loss_weight: Weight to give to loss of pre-network
            predictions. (default: 0.0)
        seed (int): Master seed. Passed in from Manager.

    Usage:
        model = WarehouseEmulationModel(...)

        # Add inputs, objectives, measures to use for training.
        model.add(data)

        # Training hyperparameters should be passed in at initialization.
        model.train()

        # Ask for objectives and measures.
        model.predict(...)
    """
    def __init__(
        self,
        network_type: type = gin.REQUIRED,
        prediction_type: str = gin.REQUIRED,
        train_epochs: int = gin.REQUIRED,
        train_batch_size: int = gin.REQUIRED,
        train_sample_size: int = None,
        train_sample_type: str = "recent",
        archive_dims: List = None,
        archive_ranges: List[List] = None,
        pre_network_type: type = None,
        pre_network_loss_func: Union[str, callable] = "ce",
        pre_repair_network_loss_func: Union[str, callable] = "ce",
        pre_network_loss_weight: float = 1.0,
        seed: int = None,
    ):
        if prediction_type not in ["regression", "classification"]:
            raise NotImplementedError(
                "Regression and classification are the only supported "
                "prediction types")

        if prediction_type == "classification":
            if archive_dims is None or archive_ranges is None:
                raise ValueError(
                    "Archive dimensions and ranges should be specified")

            if len(archive_dims) != len(archive_ranges):
                raise ValueError(
                    "Archive dimensions and ranges should have the same length")

            # TODO: Find a good place for objective classification values
            self.class_boundaries = [torch.tensor([0.5], device=DEVICE)]
            self.class_values = [np.array([0, 1])]
            for d, r in zip(archive_dims, archive_ranges):
                lb, ub = r
                boundaries = np.linspace(lb, ub, d + 1)
                self.class_boundaries.append(
                    torch.tensor(boundaries[1:-1], device=DEVICE))
                # Use the middle value of the bin as the class value
                self.class_values.append(
                    np.array([
                        np.mean([i, j])
                        for i, j in zip(boundaries[:-1], boundaries[1:])
                    ]))

        self.rng = np.random.default_rng(seed)

        self.network = network_type().to(DEVICE)  # Args handled by gin.
        params = list(self.network.parameters())
        if pre_network_type is None:
            self.pre_network = None
            self.dataset = Buffer(seed=seed)
        else:
            self.pre_network = pre_network_type().load_from_saved_weights().to(
                DEVICE)
            if isinstance(self.pre_network, WarehouseAugResnetOccupancy):
                self.dataset = AugBuffer(seed=seed)
            elif isinstance(self.pre_network,
                            WarehouseAugResnetRepairedMapAndOccupancy):
                self.dataset = DoubleAugBuffer(seed=seed)
            else:
                raise ValueError("Unknown pre-network type: %s" %
                                 type(self.pre_network))
            params += list(self.pre_network.parameters())

        self.optimizer = torch.optim.Adam(params)
        self.prediction_type = prediction_type
        self.pre_network_loss_func = pre_network_loss_func
        self.pre_network_loss_weight = pre_network_loss_weight
        self.pre_repair_network_loss_func = pre_repair_network_loss_func

        self.train_epochs = train_epochs
        self.train_batch_size = train_batch_size
        self.train_sample_size = train_sample_size
        self.train_sample_type = train_sample_type

        self.torch_rng = torch.Generator("cpu")  # Required to be on CPU.
        self.torch_rng.manual_seed(seed)

    def add(self, e: Experience):
        """Adds experience to the buffer."""
        self.dataset.add(e)

    def train(self, end_to_end=False):
        """Trains for self.train_epochs epochs on the entire dataset."""
        if len(self.dataset) == 0:
            logger.warning("Skipping training as dataset is empty")
            return

        self.network.train()
        if self.pre_network is not None:
            self.pre_network.train()

        if self.prediction_type == "classification":
            loss_calc = torch.nn.CrossEntropyLoss()
        else:
            loss_calc = torch.nn.MSELoss()  # May be configurable in the future.

        if self.pre_network is not None:
            if self.pre_network_loss_func == "ce":
                pre_loss_func = torch.nn.CrossEntropyLoss()
            elif self.pre_network_loss_func == "mse":
                pre_loss_func = torch.nn.MSELoss()
            elif self.pre_network_loss_func == "KL":
                pre_loss_func = torch.nn.KLDivLoss(reduction="batchmean")
            else:
                pre_loss_func = self.pre_network_loss_func()

            if self.pre_repair_network_loss_func == "ce":
                pre_repair_loss_func = torch.nn.CrossEntropyLoss()
            elif self.pre_repair_network_loss_func == "mse":
                pre_repair_loss_func = torch.nn.MSELoss()
            elif self.pre_repair_network_loss_func == "KL":
                pre_repair_loss_func = torch.nn.KLDivLoss()
            else:
                pre_repair_loss_func = self.pre_repair_network_loss_func()

        dataloader = self.dataset.to_dataloader(self.train_batch_size,
                                                self.torch_rng,
                                                self.train_sample_size,
                                                self.train_sample_type)
        logger.info(f"Using {len(dataloader.dataset)} samples to train.")

        actual_train_epoch = self.train_epochs
        # actual_train_sample_size = self.train_sample_size \
        #     if self.train_sample_size is not None else len(self.dataset)

        # actual_train_epoch += (actual_train_sample_size // 2000) * 5

        # # Change train epoch based on amount of data
        # if 2000 <= actual_train_sample_size < 4000:
        #     actual_train_epoch += 10
        # elif 4000 <= actual_train_sample_size < 6000:
        #     actual_train_epoch += 20
        # elif 6000 <= actual_train_sample_size < 8000:
        #     actual_train_epoch += 30
        # elif 8000 <= actual_train_sample_size < 10000:
        #     actual_train_epoch += 40
        # elif actual_train_sample_size >= 10000:
        #     actual_train_epoch += 50


        all_epoch_loss = []
        all_epoch_pre_loss = []
        all_epoch_repair_loss = []

        # Train end to end
        if end_to_end:
            for epoch in range(actual_train_epoch):
                epoch_loss = 0.0
                epoch_pre_loss = 0.0
                epoch_repair_loss = 0.0

                for sample in dataloader:
                    if self.pre_network is None:
                        inputs, objs, measures = sample
                    elif isinstance(self.pre_network,
                                    WarehouseAugResnetOccupancy):
                        inputs, objs, measures, occupancy = sample
                    elif isinstance(self.pre_network,
                                    WarehouseAugResnetRepairedMapAndOccupancy):
                        inputs, objs, measures, occupancy, repaired_map = sample
                    else:
                        raise ValueError("Unknown pre-network type: %s" %
                                         type(self.pre_network))

                    loss = self.compute_obj_measure_loss(
                        inputs,
                        objs,
                        measures,
                        loss_calc,
                    )
                    epoch_loss += loss.item()

                    if self.pre_network is not None:
                        if isinstance(self.pre_network,
                                      WarehouseAugResnetOccupancy):
                            pred_occupancy = self.pre_network.int_to_logits(
                                inputs)
                            pre_loss = pre_loss_func(
                                nn.Flatten()(pred_occupancy),
                                nn.Flatten()(occupancy))

                        elif isinstance(
                                self.pre_network,
                                WarehouseAugResnetRepairedMapAndOccupancy):
                            # Preprocess goundtruth repaired map
                            _, lvl_height, lvl_width = repaired_map.shape
                            repaired_map = int_preprocess(
                                repaired_map,
                                self.pre_network.i_size,
                                self.pre_network.nc,
                                kiva_obj_types.index("."),
                            )[:, :, :lvl_height, :lvl_width]

                            # Calculate loss on both repaired map and occupancy
                            # grid
                            pred_repaired_map, pred_occupancy = \
                                self.pre_network.int_to_logits(inputs)
                            assert pred_repaired_map.shape == repaired_map.shape
                            if self.pre_network_loss_func == "KL":
                                pred_occupancy = F.log_softmax(
                                    nn.Flatten()(pred_occupancy), dim=1)
                            pre_loss = pre_loss_func(
                                nn.Flatten()(pred_occupancy),
                                nn.Flatten()(occupancy),
                            )
                            if self.pre_repair_network_loss_func == "KL":
                                # Pass input through log_softmax while using KL
                                # loss
                                pred_repaired_map = F.log_softmax(
                                    pred_repaired_map)
                            repair_loss = pre_repair_loss_func(
                                pred_repaired_map,
                                repaired_map,
                            )

                        else:
                            raise ValueError("Unknown pre-network type: %s" %
                                             type(self.pre_network))

                        loss += self.pre_network_loss_weight * (pre_loss +
                                                                repair_loss)
                        epoch_pre_loss += pre_loss.item()
                        epoch_repair_loss += repair_loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                logger.info(f"Epoch: {epoch}")
                logger.info("Epoch Loss: %f", epoch_loss)
                logger.info(f"Pre Loss: {epoch_pre_loss}")
                logger.info(f"Repair Loss: {epoch_repair_loss}")

                all_epoch_loss.append(epoch_loss)
                all_epoch_pre_loss.append(epoch_pre_loss)
                all_epoch_repair_loss.append(epoch_repair_loss)

        # Train each part of the model separately
        else:
            if self.pre_network is not None:
                if isinstance(self.pre_network,
                              WarehouseAugResnetRepairedMapAndOccupancy):
                    # Train repair part of pre network.
                    # Freeze all params except for the repair network
                    freeze_params(self.network)
                    freeze_params(self.pre_network.occupancy_pred_mdl)

                    # Unfreeze repair pre network
                    unfreeze_params(self.pre_network.repaired_map_pred_mdl)

                    for epoch in range(actual_train_epoch):
                        epoch_repair_loss = 0.0

                        for sample in dataloader:
                            inputs, _, _, _, repaired_map = sample

                            # Convert repaired map
                            _, lvl_height, lvl_width = repaired_map.shape
                            repaired_map = int_preprocess(
                                repaired_map,
                                self.pre_network.i_size,
                                self.pre_network.nc,
                                kiva_obj_types.index("."),
                            )[:, :, :lvl_height, :lvl_width]

                            # Calculate predicted repaired map
                            pred_repaired_map, _ = \
                                self.pre_network.int_to_logits(inputs)
                            assert pred_repaired_map.shape == repaired_map.shape

                            # Calculate loss
                            if self.pre_repair_network_loss_func == "KL":
                                # Pass input through log_softmax while using KL
                                # loss.
                                pred_repaired_map = F.log_softmax(
                                    pred_repaired_map)
                            repair_loss = pre_repair_loss_func(
                                pred_repaired_map,
                                repaired_map,
                            )

                            epoch_repair_loss += repair_loss.item()

                            self.optimizer.zero_grad()
                            repair_loss.backward()
                            self.optimizer.step()

                        logger.info(f"Epoch: {epoch}")
                        logger.info(f"Repair Loss: {epoch_repair_loss}")
                        all_epoch_repair_loss.append(epoch_repair_loss)

                    # Train occupancy part of pre network.
                    freeze_params(self.pre_network.repaired_map_pred_mdl)
                    unfreeze_params(self.pre_network.occupancy_pred_mdl)
                    for epoch in range(actual_train_epoch):
                        epoch_pre_loss = 0.0
                        for sample in dataloader:
                            inputs, objs, measures, occupancy, repaired_map = sample
                            _, pred_occupancy = \
                                self.pre_network.int_to_logits(inputs)
                            if self.pre_network_loss_func == "KL":
                                pred_occupancy = F.log_softmax(
                                    nn.Flatten()(pred_occupancy), dim=1)
                            pre_loss = pre_loss_func(
                                nn.Flatten()(pred_occupancy),
                                nn.Flatten()(occupancy),
                            )

                            epoch_pre_loss += pre_loss.item()

                            self.optimizer.zero_grad()
                            pre_loss.backward()
                            self.optimizer.step()
                        logger.info(f"Epoch: {epoch}")
                        logger.info(f"Pre Loss: {epoch_pre_loss}")
                        all_epoch_pre_loss.append(epoch_pre_loss)

                    unfreeze_params(self.pre_network.repaired_map_pred_mdl)
                    unfreeze_params(self.network)

                elif isinstance(self.pre_network, WarehouseAugResnetOccupancy):
                    freeze_params(self.network)
                    unfreeze_params(self.pre_network.model)
                    for epoch in range(actual_train_epoch):
                        epoch_pre_loss = 0.0
                        for sample in dataloader:
                            inputs, objs, measures, occupancy = sample
                            pred_occupancy = self.pre_network.int_to_logits(
                                inputs)
                            pre_loss = pre_loss_func(
                                nn.Flatten()(pred_occupancy),
                                nn.Flatten()(occupancy))

                            epoch_pre_loss += pre_loss.item()

                            self.optimizer.zero_grad()
                            pre_loss.backward()
                            self.optimizer.step()

                        logger.info(f"Epoch: {epoch}")
                        logger.info(f"Pre Loss: {epoch_pre_loss}")
                    unfreeze_params(self.network)

                else:
                    raise ValueError("Unknown pre-network type: %s" %
                                     type(self.pre_network))

            # Train the main network
            freeze_params(self.pre_network)
            unfreeze_params(self.network)

            for epoch in range(actual_train_epoch):
                epoch_loss = 0.0

                for sample in dataloader:
                    if self.pre_network is None:
                        inputs, objs, measures = sample
                    elif isinstance(self.pre_network,
                                    WarehouseAugResnetOccupancy):
                        inputs, objs, measures, occupancy = sample
                    elif isinstance(self.pre_network,
                                    WarehouseAugResnetRepairedMapAndOccupancy):
                        inputs, objs, measures, occupancy, repaired_map = sample
                    else:
                        raise ValueError("Unknown pre-network type: %s" %
                                         type(self.pre_network))

                    loss = self.compute_obj_measure_loss(
                        inputs,
                        objs,
                        measures,
                        loss_calc,
                    )
                    epoch_loss += loss.item()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                logger.info(f"Epoch: {epoch}")
                logger.info("Epoch Loss: %f", epoch_loss)
                all_epoch_loss.append(epoch_loss)

            unfreeze_params(self.pre_network)

        return all_epoch_loss, all_epoch_pre_loss, all_epoch_repair_loss

    def predict_with_grad(self, inputs: torch.Tensor):
        """Same as predict() but does not convert to/from numpy."""
        pred_occupancy = None
        if self.pre_network is not None:
            if isinstance(self.pre_network, WarehouseAugResnetOccupancy):
                pred_occupancy = self.pre_network.int_to_no_crop(inputs)
                network_inputs = inputs
            elif isinstance(self.pre_network,
                            WarehouseAugResnetRepairedMapAndOccupancy):
                pred_repaired_map, pred_occupancy = \
                    self.pre_network.int_to_map_no_crop(inputs)
                network_inputs = pred_repaired_map
        return self.network.predict_objs_and_measures(network_inputs,
                                                      pred_occupancy)

    def predict(self, inputs: np.ndarray):
        """Predicts objectives and measures for a batch of solutions.

        Args:
            inputs (np.ndarray): Batch of solutions to predict.
        Returns:
            Batch of objectives and batch of measures.
        """
        self.network.eval()
        if self.pre_network is not None:
            self.pre_network.eval()

        # Handle no_grad here since we expect everything to be numpy arrays.
        with torch.no_grad():
            objs, *measures = self.predict_with_grad(
                torch.as_tensor(inputs, device=DEVICE))

            if self.prediction_type == "classification":
                obj_classes = torch.argmax(objs, dim=1)
                measure_classes = [torch.argmax(m, dim=1) for m in measures]
                objs_np = self.class_values[0][
                    obj_classes.cpu().detach().numpy()]
                measures_np = np.vstack([
                    self.class_values[1 + i][m_class.cpu().detach().numpy()]
                    for i, m_class in enumerate(measure_classes)
                ]).T
            else:
                objs_np = objs.cpu().detach().numpy().flatten()
                measures_np = np.hstack(
                    [m.cpu().detach().numpy() for m in measures])
            return objs_np, measures_np

    def compute_obj_measure_loss(self, inputs, objs, measures, loss_calc):
        """
        Pass the inputs through the entire network and calculate the final
        obj and measure loss.
        """
        pred_objs, *pred_measures = self.predict_with_grad(inputs)
        measure_dim = measures.size(dim=1)

        if self.prediction_type == "classification":
            obj_classes = torch.bucketize(objs, self.class_boundaries[0])
            measure_classes = [
                torch.bucketize(measures[:, i], self.class_boundaries[1 + i])
                for i in range(measure_dim)
            ]
            obj_loss = loss_calc(pred_objs, obj_classes)
            measure_losses = [
                loss_calc(pred_measures[i], measure_classes[i])
                for i in range(measure_dim)
            ]
        else:
            obj_loss = loss_calc(pred_objs.flatten(), objs)
            measure_losses = [
                loss_calc(pred_measures[i].flatten(), measures[:, i])
                for i in range(measure_dim)
            ]

        # TODO: Maybe add weighting
        loss = (obj_loss + sum(measure_losses)) / (1 + measure_dim)
        return loss

    def save(self, pickle_path: Path, pytorch_path: Path):
        """Saves data to a pickle file and a PyTorch file.

        The PyTorch file holds the network and the optimizer, and the pickle
        file holds the rng and the dataset. See here for more info:
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
        """
        logger.info("Saving WarehouseEmulationModel pickle data")
        with pickle_path.open("wb") as file:
            cloudpickle.dump(
                {
                    "rng": self.rng,
                    "dataset": self.dataset,
                },
                file,
            )

        logger.info("Saving WarehouseEmulationModel PyTorch data")
        state_dict = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "torch_rng": self.torch_rng.get_state(),
        }

        if self.pre_network is not None:
            state_dict["pre_network"] = self.pre_network.state_dict()

        torch.save(state_dict, pytorch_path)

    def load(self, pickle_path: Path, pytorch_path: Path, map_location=None):
        """Loads data from files saved by save()."""
        with open(pickle_path, "rb") as file:
            pickle_data = pkl.load(file)
            self.rng = pickle_data["rng"]
            self.dataset = pickle_data["dataset"]

        pytorch_data = torch.load(pytorch_path, map_location=map_location)
        self.network.load_state_dict(pytorch_data["network"])
        if self.pre_network is not None:
            self.pre_network.load_state_dict(pytorch_data["pre_network"])
        self.optimizer.load_state_dict(pytorch_data["optimizer"])
        self.torch_rng.set_state(pytorch_data["torch_rng"])
        return self
