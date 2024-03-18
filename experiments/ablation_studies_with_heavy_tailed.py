import logging
import sys
from datetime import datetime

from pathlib import Path

from ovg_experiments.ablation_common import (
    ablation_experiment,
    AblationStudyConfig,
    set_seed,
)

from ovg_experiments.datagen_settings import DataGenSettings


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("ablation-studies")

    datagen_settings = DataGenSettings.get_default()

    config = AblationStudyConfig.get_testing()
    # config = _config.get_default()

    datagen_settings.num_samples = config.num_samples
    heavy_tailed = True

    seed = 42
    set_seed(42)

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

    ablation_experiment(
        results_dir,
        datagen_settings,
        config.num_runs,
        config.lrs,
        config.hidden_sizes,
        config.epochs,
        heavy_tailed,
    )
