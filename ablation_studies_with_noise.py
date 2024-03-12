import numpy as np
import yaml
from data.generate_data import ablation_generation
from predictors import MarginalPredictor, ProposedPredictor, OptimalPredictor, ImputedPredictor
from evaluation import compute_zero_shot_loss
from utils.plot_utils import *
import random
import torch
from itertools import product

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def run_experiment(dataset, lr=0.01, hidden_size=64, num_epochs = 50):

    reference_predictor = OptimalPredictor()
    proposed_predictor = ProposedPredictor(lr = lr, hidden_size = hidden_size, num_epochs = num_epochs)
    marginal_predictor = MarginalPredictor()
    imputation_predictor = ImputedPredictor()

    for pred in [reference_predictor, proposed_predictor, marginal_predictor, imputation_predictor]:
        pred.fit(dataset.data_source, dataset.data_target)


    predictors_dict = {"Proposed": proposed_predictor}
    predictors_dict["Oracle"] = reference_predictor
    predictors_dict["Marginal"] = marginal_predictor
    predictors_dict["Imputation"] = imputation_predictor

    loss = compute_zero_shot_loss(
        reference_predictor, predictors_dict, dataset.data_target, num_samples=1000, systematic=True
    )

    return loss

def ablation_studies(base_path,  noise, num_runs=5, mode='linear', lr = 0.01, hidden_size = 64, epoch = 50):
    set_seed()
    with open(join(base_path, "configs.yml"), "r") as f:
        settings = yaml.safe_load(f)
    settings['dataset']['num_samples'] = 10000

    results = {'level0': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []},
               'level1': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []},
               'level2': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []}}

    for run in range(num_runs):
        for level in [0, 1, 2]:
            dataset = ablation_generation(base_path, settings['dataset'], level, mode, noise=noise)
            loss = run_experiment(dataset, lr =lr, hidden_size=hidden_size, num_epochs=epoch)
            results['level'+str(level)]['proposed'].append(loss['Proposed'])
            results['level' + str(level)]['marginal'].append(loss['Marginal'])
            results['level' + str(level)]['oracle'].append(loss['Oracle'])
            results['level' + str(level)]['imputation'].append(loss['Imputation'])

    return results

def calculate_mean_summary_table(summary_noises, modes, noises, predictors, levels):
    mean_table = {str(n): {mode: {level: {predictor: {'mean': [], 'std': []} for predictor in predictors} for level in levels} for mode in modes} for n in noises}
    perc_increase_table = {str(n): {mode: {level: {predictor: {'perc': []} for predictor in predictors} for level in levels} for mode in modes} for n in noises}
    for n in noises:
        for mode in modes:
            for key in summary_noises[str(n)][mode].keys():
                for predictor in predictors:
                    mean = np.mean(summary_noises[str(n)][mode][key][predictor])
                    std = np.std(summary_noises[str(n)][mode][key][predictor])
                    mean_table[str(n)][mode][key][predictor]['mean'] = mean
                    mean_table[str(n)][mode][key][predictor]['std'] = std
                for predictor in predictors:
                    oracle_mean = mean_table[str(n)][mode][key]['oracle']['mean']
                    cur_mean = mean_table[str(n)][mode][key][predictor]['mean']
                    perc_increase_table[str(n)][mode][key][predictor]['perc'] = (cur_mean - oracle_mean)/oracle_mean
    print('mean_table', mean_table)


    return mean_table, perc_increase_table

def main():

    lrs = [0.01]
    hidden_sizes = [64]
    epochs = [50]
    noises = [0.01, 0.2, 0.4, 0.6, 0.8, 1]
    predictors = ['proposed', 'marginal', 'oracle', 'imputation']
    levels = ['level0', 'level1', 'level2']
    modes = ['linear'] #['linear', 'general']

    summary_noises = {str(n): {mode: None} for mode in modes for n in noises}
    for n in noises:
        for lr, hidden_size, epoch in product(lrs, hidden_sizes, epochs):
            for mode in modes:
                base_path = "experiments"
                summary_noises[str(n)][mode] = ablation_studies(base_path, n, mode = mode, lr = lr, hidden_size = hidden_size, epoch = epoch)
                print(f"learning rate {lr}, hidden_size {hidden_size}, num of epochs {epoch}, noise {n}, mode {mode}")
                print('result:', summary_noises[str(n)][mode])

    save_path = "experiments/ablation"
    np.save(save_path + '/summary_noises.npy', summary_noises)

    mean_table = calculate_mean_summary_table(summary_noises, modes, noises, predictors, levels)
    np.save(save_path + '/mean_table_noises.npy', mean_table)



if __name__ == "__main__":
   main()

