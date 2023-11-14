#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

class CrossValidation:
    def __init__(self, n_splits=5,
                 save_train_test_split = True, random_state= RANDOM_STATE,
                 save_path='cross_validation_datasets'):
        self.n_splits = n_splits
        self.save_path = save_path
        self.random_state= random_state
        self.save_train_test_split = save_train_test_split
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def save_splits(self, splits, save_split_file):
        with open(os.path.join(self.save_path, save_split_file), 'wb') as f:
            pickle.dump(splits, f)

    def load_splits(self, save_split_file):
        with open(os.path.join(self.save_path, save_split_file), 'rb') as f:
            splits = pickle.load(f)
        return splits

    def get_label_percentage(self, df, label_col='Review'):
        count = df[label_col].value_counts()
        percentage = df[label_col].value_counts(normalize=True) * 100
        result = pd.concat([count, percentage], axis=1).round(2)
        result.columns = ['Count', 'Percentage']
        return result


    def split(self, df,
              label_col='Review',
              save_split_file='cross_validation_splits.pkl'):

        splits = []
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        label_distributions = {}
        for i, (train_index, test_index) in enumerate(skf.split(df, df[label_col])):
            train_df = df.loc[train_index].copy()
            test_df = df.loc[test_index].copy()
            train_label_percentage = self.get_label_percentage(train_df)
            label_distributions[f'train_{i}'] = train_label_percentage
            test_label_percentage = self.get_label_percentage(test_df)
            label_distributions[f'test_{i}'] = test_label_percentage
            if self.save_train_test_split:
                train_filename = f'{self.save_path}/train_{i}.csv'
                test_filename = f'{self.save_path}/test_{i}.csv'
                train_df.to_csv(train_filename, index = False)
                test_df.to_csv(test_filename,  index = False)
            splits.append((train_index, test_index))

        self.save_splits(splits, save_split_file)
        label_dist_df = pd.concat(label_distributions, names=['Dataset', 'Label'])
        label_dist_df.to_csv(os.path.join(self.save_path, 'label_distribution.csv'))

        return splits

