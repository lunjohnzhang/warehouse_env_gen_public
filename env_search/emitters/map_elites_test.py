import fire
import numpy as np

from src.archives import GridArchive
from src.emitters.map_elites_baseline_emitter import MapElitesBaselineEmitter


def main():
    archive = GridArchive(dims=[2, 2],
                          ranges=[[0, 1], [0, 1]],
                          record_history=False)
    sol_dim = 16
    archive.initialize(sol_dim)
    me_emitter = MapElitesBaselineEmitter(archive,
                                          np.zeros(sol_dim),
                                          batch_size=2,
                                          num_objects=2,
                                          initial_population=5,
                                          mutation_k=10)

    for _ in range(5):
        sols = me_emitter.ask()
        print(f"Returned solutions: {sols}")

        objs = np.random.random(size=len(sols))
        measures = np.random.random(size=(len(sols), 2))
        print(f"Telling objs = {objs}, measures = {measures}")
        me_emitter.tell(sols, objective_values=objs, behavior_values=measures)

    print(archive.stats)


if __name__ == '__main__':
    fire.Fire(main)
