import os
import gin
import pickle as pkl
import fire
from env_search.utils.logging import setup_logging
from logdir import LogDir
import matplotlib.pyplot as plt
import shutil

# Including this makes gin config work because main imports (pretty much)
# everything.
import env_search.main  # pylint: disable = unused-import
from env_search.warehouse.emulation_model.emulation_model import WarehouseEmulationModel


def plot_loss(
    logdir,
    all_epoch_loss,
    all_epoch_pre_loss,
    all_epoch_repair_loss,
):
    fig, (loss_ax, pre_loss_ax, repair_loss_ax) = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    loss_ax.plot(all_epoch_loss)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Obj + Measure Loss")
    loss_ax.set(xlim=(0, None), ylim=(0, 50))

    pre_loss_ax.plot(all_epoch_pre_loss)
    pre_loss_ax.set_xlabel("Epoch")
    pre_loss_ax.set_ylabel("Tile usage Loss")
    pre_loss_ax.set(xlim=(0, None), ylim=(0, 1))

    repair_loss_ax.plot(all_epoch_repair_loss)
    repair_loss_ax.set_xlabel("Epoch")
    repair_loss_ax.set_ylabel("Repair Loss")
    repair_loss_ax.set(xlim=(0, None), ylim=(0, 50))

    fig.savefig(logdir.file("Loss.png"), dpi=300, bbox_inches="tight")


def offline_train(
    log_dir_data: str,
    warehouse_config: str,
    seed: int = 0,
    end_to_end: bool = False,
    data_mode: str = "small", # "small" or "large"
):
    setup_logging(on_worker=False)
    gin.clear_config()
    gin.parse_config_file(warehouse_config)

    reload_em_pkl = os.path.join(log_dir_data, "reload_em.pkl")
    with open(reload_em_pkl, "rb") as f:
        data = pkl.load(f)
        dataset = data["dataset"]

    # Save train model
    exp_name = gin.query_parameter("experiment.name")
    if end_to_end:
        exp_name += "_end-to-end"
    logdir = LogDir(exp_name, rootdir="./logs", uuid=True)

    # Save configuration options.
    with logdir.pfile("config.gin").open("w") as file:
        file.write(gin.config_str(max_line_length=120))

    # Write the seed.
    with logdir.pfile("seed").open("w") as file:
        file.write(str(seed))

    emulation_model = WarehouseEmulationModel(seed=seed + 420)
    if data_mode == "small":
        emulation_model.train_sample_size = 1000
    elif data_mode == "large":
        emulation_model.train_sample_size = None # Use all data
    emulation_model.dataset = dataset
    (
        all_epoch_loss,
        all_epoch_pre_loss,
        all_epoch_repair_loss,
    ) = emulation_model.train(end_to_end=end_to_end)

    # Save model
    emulation_model.save(logdir.pfile("reload_em-tmp.pkl"),
                         logdir.pfile("reload_em-tmp.pth"))

    logdir.pfile("reload_em-tmp.pkl").rename(logdir.pfile("reload_em.pkl"))
    logdir.pfile("reload_em-tmp.pth").rename(logdir.pfile("reload_em.pth"))

    plot_loss(
        logdir,
        all_epoch_loss,
        all_epoch_pre_loss,
        all_epoch_repair_loss,
    )


if __name__ == "__main__":
    fire.Fire(offline_train)