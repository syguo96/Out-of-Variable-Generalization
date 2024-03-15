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


def _main(
    result_dir: Path,
    data_generation_settings: DataGenSettings,
    num_runs,  # 5
    lrs,
    hidden_sizes,
    epochs,
):
    logger = logging.getLogger("ablation-studies")

    mode_results: Dict[Mode, List[AblationStudyResults]] = {mode: [] for mode in Mode}

    for mode in (Mode.linear, Mode.general):
        for lr, hidden_size, epoch in product(lrs, hidden_sizes, epochs):
            r: AblationStudyResults = ablation_studies(
                data_generation_settings, mode, num_runs, lr, hidden_size, epoch
            )
            mode_results[mode].append(r)

    for mode, results in mode_results.items():
        summary = get_summary(results)
        np.save(result_dir / f"mean_{mode.name}.npy", summary)
        logger.info(f"\n-- results for {mode.name} --\n" + format_summary(summary))

    logger.info(f"results saved in {result_dir}")


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

    seed = 42
    set_seed(42)

    results_dir = (
        Path.cwd() / f'results_ablation_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'
    )
    results_dir.mkdir(parents=True)

    for k, v in datagen_settings.to_dict().items():
        logger.info(f"{k}:\t{v}")
    logger.info(f"seed:\t{seed}")
    logger.info(f"lrs:\t{config.lrs}")
    logger.info(f"hidden sizes\t{config.hidden_sizes}")
    logger.info(f"epochs:\t{config.epochs}")
    logger.info(f"result directory:\t{results_dir}")

    _main(
        results_dir,
        datagen_settings,
        config.num_runs,
        config.lrs,
        config.hidden_sizes,
        config.epochs,
    )
