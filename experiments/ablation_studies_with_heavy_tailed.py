import logging
import sys
from datetime import datetime

from pathlib import Path

from ovg_experiments.ablation_common import (
    ablation_experiment,
    AblationStudyConfig,
    set_seed,
)

from ovg_experiments.simulated_data import DataGenSettings


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("ablation-studies")

    heavy_tailed = True
    seed = 42
    datagen_settings = DataGenSettings(
        num_samples=10000,
        split_fraction=0.9,
        noise_var=0.1,
        noise_skew=0.0,
        noise_mean=0.0,
    )
    config = AblationStudyConfig.from_dict(
        {
            "lrs": (0.01, 0.001),
            "hidden_sizes": (64, 32),
            "epochs": (30, 50),
            "num_runs": 5,
        }
    )

    results_dir = (
        Path.cwd()
        / f'results_ablation_w_htail_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'
    )
    for k, v in datagen_settings.to_dict().items():
        logger.info(f"{k}:\t{v}")
    logger.info(f"seed:\t{seed}")
    logger.info(f"lrs:\t{config.lrs}")
    logger.info(f"hidden sizes\t{config.hidden_sizes}")
    logger.info(f"epochs:\t{config.epochs}")
    logger.info(f"result directory:\t{results_dir}")

    set_seed(seed)
    ablation_experiment(
        results_dir,
        datagen_settings,
        config.num_runs,
        config.lrs,
        config.hidden_sizes,
        config.epochs,
        heavy_tailed,
    )