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

def ablation_studies(base_path, summary, num_runs=5, mode='linear', lr = 0.01, hidden_size = 64, epoch = 50, is_heavy_tailed=True):

    set_seed()
    with open(join(base_path, "configs.yml"), "r") as f:
        settings = yaml.safe_load(f)
    settings['dataset']['num_samples'] = 10000

    results = {'level0': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []},
               'level1': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []},
               'level2': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []}}
    #summary = {'level0': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []},
    #           'level1': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []},
    #           'level2': {'proposed': [], 'marginal': [], 'oracle': [], 'imputation': []}}
    for run in range(num_runs):
        for level in [0, 1, 2]:
            dataset = ablation_generation(base_path, settings['dataset'], level, mode, is_heavy_tailed=is_heavy_tailed)
            loss = run_experiment(dataset, lr =lr, hidden_size=hidden_size, num_epochs=epoch)
            results['level'+str(level)]['proposed'].append(loss['Proposed'])
            results['level' + str(level)]['marginal'].append(loss['Marginal'])
            results['level' + str(level)]['oracle'].append(loss['Oracle'])
            results['level' + str(level)]['imputation'].append(loss['Imputation'])

    for key in results.keys():
        for predictor in ['proposed', 'marginal', 'oracle', 'imputation']:
            mean = np.mean(results[key][predictor])
            std = np.std(results[key][predictor])
            summary[key][predictor]['mean'].append(mean)
            summary[key][predictor]['std'].append(std)

    # Save and print
    data_path = join(base_path, 'ablation/' + mode)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    np.save(data_path + '/results.npy', results)
    np.save(data_path + '/summary.npy', summary)

    return summary

# def main():
#     base_path = "./experiments/"
#     _ = ablation_studies(base_path, num_runs = 10, mode = 'linear')
#     _ = ablation_studies(base_path, num_runs = 10, mode = 'general')
#     result1 = np.load('./experiments/ablation/linear/summary.npy', allow_pickle=True)
#     result2 = np.load('./experiments/ablation/general/summary.npy', allow_pickle=True)
#     print('linear result:', result1)
#     print('general result:', result2)

def main():
    # lrs = [0.01]
    # hidden_sizes = [64]
    # epochs = [50]
    lrs = [0.01, 0.001]
    hidden_sizes = [64, 32]
    epochs = [30, 50]
    summary_linear = {'level0': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level1': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level2': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}}}
    summary_general = {'level0': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level1': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level2': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}}}
    for lr, hidden_size, epoch in product(lrs, hidden_sizes, epochs):
        base_path = "experiments"
        print(f"learning rate {lr}, hidden_size {hidden_size}, num of epochs {epoch}")
        summary_linear = ablation_studies(base_path, summary_linear, mode = 'linear', lr = lr, hidden_size = hidden_size, epoch = epoch, is_heavy_tailed = True)
        summary_general = ablation_studies(base_path, summary_general, mode = 'general', lr = lr, hidden_size = hidden_size, epoch = epoch, is_heavy_tailed=True )
        # result1 = np.load(base_path + 'ablation/linear/summary.npy', allow_pickle=True)r
        # result2 = np.load(base_path + 'ablation/general/summary.npy', allow_pickle=True)
        print(f"learning rate {lr}, hidden_size {hidden_size}, num of epochs {epoch}")
        print('linear result:', summary_linear)
        print('general result:', summary_general)

    mean_linear = {'level0': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level1': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level2': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}}}
    mean_general = {'level0': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level1': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}},
               'level2': {'proposed': {'mean': [], 'std': []}, 'marginal': {'mean': [], 'std': []}, 'oracle': {'mean': [], 'std': []}, 'imputation': {'mean': [], 'std': []}}}

    for key in summary_linear.keys():
        for predictor in ['proposed', 'marginal', 'oracle', 'imputation']:
            mean = np.mean(summary_linear[key][predictor]['mean'])
            mean_std = np.mean(summary_linear[key][predictor]['std'])
            mean_linear[key][predictor]['mean'].append(mean)
            mean_linear[key][predictor]['std'].append(mean_std)

    for key in summary_general.keys():
        for predictor in ['proposed', 'marginal', 'oracle', 'imputation']:
            mean = np.mean(summary_general[key][predictor]['mean'])
            mean_std = np.mean(summary_general[key][predictor]['std'])
            mean_general[key][predictor]['mean'].append(mean)
            mean_general[key][predictor]['std'].append(mean_std)

    print('mean_linear', mean_linear)
    print('mean_general', mean_general)
    save_path = "experiments/ablation"
    np.save(save_path + '/mean_linear_w_ht.npy', mean_linear)
    np.save(save_path + '/mean_general_w_ht.npy', mean_general)

if __name__ == "__main__":
    main()
