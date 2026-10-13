import torch
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import importlib
import argparse
from datetime import datetime
from dataloader import data_generator1, swat, esa, other_datasets
from utils import print_object
from ts_datasets.ts_datasets.anomaly import Exathlon, SMD
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from models.MemMS.network.model import base_Model
from models.MemMS.trainer.trainer import Trainer


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--seed', default=1, type=int,
                    help='seed value')
parser.add_argument('--selected_dataset', default='SMD', type=str,
                    choices=['SMD', 'Exathlon', 'SWaT', 'ESA'],
                    help='Dataset of choice')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
args = parser.parse_args()

device = torch.device(args.device)
data_type = args.selected_dataset
method = 'MemMS'
selected_dataset = args.selected_dataset


Configs = importlib.import_module(f"conf.memms.{data_type}_Configs").Config
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed

print(f'Dataset: {data_type}')
print(f'Method:  {method}')
print(f'Random Seed:  {SEED}')

# Load datasets
if selected_dataset in ['SWaT', 'ESA']:
    model_num = 1
else:
    datasets = {'SMD': SMD, 'Exathlon': Exathlon}
    dt = datasets[selected_dataset]()
    model_num = len(dt)

# Aggregate statistics from full dataset
all_test_rpa_score, all_test_pa_score, all_test_pw_score = [], [], []
all_anomaly_num = []
all_test_aff_score, all_test_aff_precision, all_test_aff_recall = [], [], []
all_test_roc_auc, all_test_pr_auc = [], []
all_test_vus_roc, all_test_vus_pr = [], []
if model_num == 0:
    raise ValueError(f"No entities found for {selected_dataset}")
for idx in tqdm(range(model_num)):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)


    if selected_dataset == 'SWaT':
        train_data, test_data, train_labels, test_labels = swat()
    elif selected_dataset == 'ESA':
        train_data, test_data, train_labels, test_labels = esa()
    else:
        time_series, meta_data = dt[idx]
        train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta_data)

    print('>' * 32, len(train_data))
    print('>' * 32, len(test_data))

    print("Data loaded ...")

    configs.input_channels = train_data.shape[1]
    train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator1(train_data, test_data, train_labels, test_labels, SEED, configs)

    model = base_Model(configs, device).to(device)
    weight_decay = configs.weight
    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                           weight_decay=weight_decay)
    # Trainer
    test_score_origin, test_aff, test_rpa_score, test_pa_score, test_pw_score, predict, test_roc_auc, test_pr_auc, test_vus_roc, test_vus_pr = Trainer(model, model_optimizer, train_dl,
                                                                                 val_dl, test_dl, device, configs, idx, SEED)
    all_test_roc_auc.append(test_roc_auc)
    all_test_pr_auc.append(test_pr_auc)
    all_test_vus_roc.append(test_vus_roc)
    all_test_vus_pr.append(test_vus_pr)
    all_anomaly_num.append(test_anomaly_window_num)
    all_test_aff_precision.append(test_aff["precision"])
    all_test_aff_recall.append(test_aff["recall"])
    all_test_aff_score.append(test_aff)
    all_test_rpa_score.append(test_rpa_score)
    all_test_pa_score.append(test_pa_score)
    all_test_pw_score.append(test_pw_score)


all_anomaly_num = np.array(all_anomaly_num)
sum_anomaly_num = np.sum(all_anomaly_num)
all_test_aff_precision = np.array(all_test_aff_precision)
all_test_aff_precision = (all_test_aff_precision * all_anomaly_num / sum_anomaly_num
                          if sum_anomaly_num else np.zeros_like(all_test_aff_precision))
test_aff_precision = np.nansum(all_test_aff_precision)
all_test_aff_recall = np.array(all_test_aff_recall)
all_test_aff_recall = (all_test_aff_recall * all_anomaly_num / sum_anomaly_num
                       if sum_anomaly_num else np.zeros_like(all_test_aff_recall))
test_aff_recall = np.nansum(all_test_aff_recall)
if test_aff_precision + test_aff_recall == 0:
    test_aff_f1 = 0
else:
    test_aff_f1 = 2 * (test_aff_precision * test_aff_recall) / (test_aff_precision + test_aff_recall)

total_test_rpa_score = sum(all_test_rpa_score, ScoreAcc())
total_test_pa_score = sum(all_test_pa_score, ScoreAcc())
total_test_pw_score = sum(all_test_pw_score, ScoreAcc())
print('>' * 32)
print("affiliation metrics:\n",
      f"Precision: {test_aff_precision:.5f}\n",
      f"Recall:    {test_aff_recall:.5f}\n"
      f"F1 Scores: {test_aff_f1:.5f}\n"
      "Revised-point-adjusted metrics:\n",
      f"Precision: {total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
      f"Recall:    {total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n"
      f"F1 score:  {total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n",
      "Point-adjusted metrics:\n",
      f"Precision: {total_test_pa_score.precision(ScoreType.PointAdjusted):.5f}\n",
      f"Recall:    {total_test_pa_score.recall(ScoreType.PointAdjusted):.5f}\n",
      f"F1 score:  {total_test_pa_score.f1(ScoreType.PointAdjusted):.5f}\n",
      "Point-wise metrics:\n",
      f"Precision: {total_test_pw_score.precision(ScoreType.Pointwise):.5f}\n",
      f"Recall:    {total_test_pw_score.recall(ScoreType.Pointwise):.5f}\n"
      f"F1 Scores:  {total_test_pw_score.f1(ScoreType.Pointwise):.5f}\n"
      "NAB Scores:\n",
      f"NAB Score (balanced):       {total_test_pa_score.nab_score():.5f}\n",
      f"NAB Score (high precision): {total_test_pa_score.nab_score(fp_weight=0.22):.5f}\n",
      f"NAB Score (high recall):    {total_test_pa_score.nab_score(fn_weight=2.0):.5f}\n"
      "seed:", SEED, "\n"
      "config setup:\n"
      )
str_conf = print_object(configs)
if hasattr(configs, 'augmentation'):
    str_aug = print_object(configs.augmentation)
    str_conf += '\n' + str_aug

train_time = datetime.now()-start_time
print(f"Training time is : {train_time}")

path = "./results"
if not os.path.exists(path):
    os.makedirs(path)

summary = os.path.join("results", f"{method}_{selected_dataset}_summary.csv")
if os.path.exists(summary):
    df = pd.read_csv(summary, index_col=0)
else:
    df = pd.DataFrame()
model_name = method + f"{df.shape[1]}"

df.loc["Hyper-parameter", model_name] = str_conf
df.loc["seed", model_name] = SEED
df.loc["Train Time", model_name] = train_time
df.loc["Affiliation Precision", model_name] = round(test_aff_precision, 5)
df.loc["Affiliation Recall", model_name] = round(test_aff_recall, 5)
df.loc["Affiliation F1", model_name] = round(test_aff_f1, 5)
df.loc["RPA Precision", model_name] = round(total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA Recall", model_name] = round(total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA F1", model_name] = round(total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted), 5)
df.loc["PA Precision", model_name] = round(total_test_pa_score.precision(ScoreType.PointAdjusted), 5)
df.loc["PA Recall", model_name] = round(total_test_pa_score.recall(ScoreType.PointAdjusted), 5)
df.loc["PA F1", model_name] = round(total_test_pa_score.f1(ScoreType.PointAdjusted), 5)
df.loc["Point-wise Precision", model_name] = round(total_test_pw_score.precision(ScoreType.Pointwise), 5)
df.loc["Point-wise Recall", model_name] = round(total_test_pw_score.recall(ScoreType.Pointwise), 5)
df.loc["Point-wise F1", model_name] = round(total_test_pw_score.f1(ScoreType.Pointwise), 5)
df.loc["NAB Score (balanced)", model_name] = round(total_test_rpa_score.nab_score(), 5)
df.loc["NAB Score (high precision)", model_name] = round(total_test_rpa_score.nab_score(fp_weight=0.22), 5)
df.loc["NAB Score (high recall)", model_name] = round(total_test_rpa_score.nab_score(fn_weight=2.0), 5)


# Continuous metrics are averaged over entities with available values.
for metric, values in [
    ("Test ROC-AUC", all_test_roc_auc),
    ("Test PR-AUC", all_test_pr_auc),
    ("Test VUS-ROC", all_test_vus_roc),
    ("Test VUS-PR", all_test_vus_pr),
]:
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    value = float(valid.mean()) if len(valid) else np.nan
    print(f"{metric}: {value:.5f}")
    df.loc[metric, model_name] = round(value, 5)

df.to_csv(summary, index=True)
print(f"Results saved to: {summary}")
