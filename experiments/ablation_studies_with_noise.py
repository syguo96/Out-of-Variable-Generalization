import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from ovg_experiments.ablation_common import (AblationStudyResults, Mode,
                                             ablation_studies, format_summary,
                                             set_seed)
from ovg_experiments.simulated_data import DataGenSettings


def _main(
    noises: Iterable[float],
    datagen_settings: DataGenSettings,
    num_runs: int,
    lr: float,
    hidden_size: int,
    epoch: int,
) -> None:
    results: Dict[str, Dict[Mode, AblationStudyResults]] = {
        str(noise): {mode: AblationStudyResults() for mode in Mode} for noise in noises
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
            logger.info(
                "\nresults:\n" + format_summary(result.summary_dict())  # type: ignore
            )
            results[str(noise)][mode] = result

    raw_results = {
        str(noise): {mode: results[str(noise)][mode].raw() for mode in Mode}
        for noise in noises
    }
    np.save(results_dir / "summary_noises.npy", raw_results)  # type: ignore

    mean_table = {
        str(noise): {
            mode: results[str(noise)][mode].summary_dict(with_perc=True)
            for mode in Mode
        }
        for noise in noises
    }
    np.save(results_dir / "mean_table_noises.npy", mean_table)  # type: ignore


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger("ablation-studies-with-noise")

    results_dir = (
        Path.cwd()
        / f'results/ablation/'
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = 42
    num_runs = 5
    lr = 0.01
    hidden_size = 64
    epochs = 50
    noises = [0.01, 0.2, 0.4, 0.6, 0.8, 1]

    datagen_settings = DataGenSettings(
        num_samples=10000,
        split_fraction=0.9,
        noise_var=0.1,
        noise_skew=0.0,
        noise_mean=0.0,
    )

    set_seed(seed)
    _main(noises, datagen_settings, num_runs, lr, hidden_size, epochs)
