import os
import fire
import json

from env_search.analysis.visualize_sim.plot_figure import draw_animation_with_orientation
from env_search.analysis.visualize_sim.read_file import read_tasks_file, read_paths_file
from env_search.utils import kiva_env_number2str, kiva_env_str2number


def main(log_dir: str, map_img_path: str):

    sim_results_dir = os.path.join(log_dir, "results")

    for sim_dir_f in os.listdir(sim_results_dir):
        sim_dir = os.path.join(sim_results_dir, sim_dir_f)
        with open(os.path.join(sim_dir, "config.json")) as f:
            config = json.load(f)

        simulation_time = config["simulation_time"]
        map_json = json.loads(config["map"])
        env_np = kiva_env_str2number(map_json["layout"])
        n_row, n_col = env_np.shape
        paths = read_paths_file(sim_dir, simulation_time)
        (
            num_drives,
            throughput,
            extra_cost,
            extra_costs_percentage,
            station_utility,
        ) = read_tasks_file(sim_dir, simulation_time)

        # breakpoint()
        draw_animation_with_orientation(
            paths,
            throughput,
            map_img_path,
            n_col,
            n_row,
            env_np,
        )


if __name__ == '__main__':
    fire.Fire(main)