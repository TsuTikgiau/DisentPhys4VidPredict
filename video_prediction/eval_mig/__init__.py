from .basis_mig import plot_mig, plot_mig_properties
from .move_mig import EvalMigMove
from .hit_wall_mig import EvalMigHitWall
from .collision_mig import EvalMigCollision


def get_mig_eval(dataset_type):
    mig_eval_dict = \
        {'move_mig': EvalMigMove,
         'hit_wall_mig': EvalMigHitWall,
         'collision_mig': EvalMigCollision}

    if dataset_type not in mig_eval_dict.keys():
        raise KeyError

    return mig_eval_dict[dataset_type]
