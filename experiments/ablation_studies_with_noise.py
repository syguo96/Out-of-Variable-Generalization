import pprint
import logging
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np
from ovg_experiments.ablation_common import (
    AblationStudyResults,
    AblationStudyConfig,
    Mode,
    ablation_studies,
    set_seed,
    get_summary,
    format_summary,
)
from ovg_experiments.datagen_settings import DataGenSettings


def _main(noises, num_runs: int, lr: float, hidden_size: int, epoch: int) -> None:
    results = {
        str(noise): {mode: AblationStudyResults for mode in Mode} for noise in noises
    }
    for noise in noises:
        for mode in Mode:
            result: AblationStudyResults = ablation_studies(
                datagen_settings,
                mode,
                num_runs,
                lr,
                hidden_size,
                epoch,
                noise=noise,
            )
            config_ = {
                "learning rate": lr,
                "hidden size": hidden_size,
                "epoch": epoch,
                "noise": noise,
                "mode": mode.name,
            }
            for k, v in config_.items():
                logger.info(f"{k}:\t{v}")
            logger.info("\nresults:\n" + format_summary(result.summary_dict()))
            results[str(noise)][mode] = result

    raw_results = {
        str(noise): {mode: results[str(noise)][mode].raw() for mode in Mode}
        for noise in noises
    }
    np.save(results_dir / "summary_noises.npy", raw_results)

    mean_table = {
        str(noise): {
            mode: results[str(noise)][mode].summary_dict(with_perc=True)
            for mode in Mode
        }
        for noise in noises
    }
    np.save(results_dir / "mean_table_noises.npy", mean_table)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger("ablation-studies-with-noise")

    results_dir = (
        Path.cwd()
        / f'results_ablation_with_noise_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'
    )
    results_dir.mkdir(parents=True)

    seed = 42
    num_runs = 5
    lr = 0.01
    hidden_size = 64
    epochs = 50
    noises = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    datagen_settings = DataGenSettings.get_default()
    datagen_settings.num_samples = 10000

    set_seed(seed)
    _main(noises, num_runs, lr, hidden_size, epochs)
