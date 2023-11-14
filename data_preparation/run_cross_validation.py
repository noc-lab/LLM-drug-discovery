#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import argparse
import pandas as pd
import numpy as np
from cross_validation import CrossValidation
import sys
sys.path.insert(0, './models')
from funs import setup_logger
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def save_mean_std_metrics(cv_metrics, saved_path, saved_csv):
    cv_metrics_df = pd.DataFrame(cv_metrics)
    cv_mean_metrics = cv_metrics_df.mean(axis=0).round(2)
    cv_std_metrics = cv_metrics_df.std(axis=0).round(2)
    metrics_summary = pd.concat([cv_metrics_df, cv_mean_metrics.to_frame().T, cv_std_metrics.to_frame().T], ignore_index=True)
    index_names = list(range(cv_metrics_df.shape[0])) + ['mean', 'std']
    metrics_summary.index = index_names
    metrics_summary.to_csv(os.path.join(saved_path, saved_csv), index=True)

def main():

    parser=argparse.ArgumentParser(description="Cross validation",
                                   prog = "Cross Validator",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_fold", type=int, default=5)
    parser.add_argument("--input_path",type=str, default='./data_preparation/output/datasets/Nipah/Nipah_pre_explanations.csv')
    parser.add_argument("--save_dir", type=str, default='./data_preparation/output/datasets/Nipah/cross_validation_datasets')
    parser.add_argument("--log_path", type=str, default='./data_preparation/output/datasets/Nipah/cross_validation_datasets/cross_validation.log')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    logger = setup_logger(args.log_path)
    SAVE_CROSS_DIR = os.path.join(args.save_dir, 'general')
    if not os.path.exists(SAVE_CROSS_DIR):
        os.makedirs(SAVE_CROSS_DIR)

    data = pd.read_csv(args.input_path)
    cv = CrossValidation(n_splits=args.num_fold,save_train_test_split = True, save_path=SAVE_CROSS_DIR)
    splits = cv.split(df=data)

    for i, (train_index, test_index) in enumerate(splits):
        Train = data.iloc[train_index].copy()
        Test = data.iloc[test_index].copy()

if __name__ == '__main__':
    main()





