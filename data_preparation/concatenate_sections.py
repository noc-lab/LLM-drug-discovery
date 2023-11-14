#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import argparse
import pandas as pd
import numpy as np
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def preprocess_input(df, columns_to_save):
    df['Keyword'] = df['Keyword'].replace('Missing', None)
    df['Review'] = np.where(df['Review'], 'Yes', 'No')

    df['Combined'] = df.apply(lambda row: ('Paper:\nTitle: ' + row['Title'] + (
        '\n' + row['Keyword'] if row['Keyword'] is not None else '') + '\n' + 'Abstract: ' + row['Abstract']), axis=1)
    columns_to_save.append("Combined")
    return df[columns_to_save]


def main():
    parser=argparse.ArgumentParser(description = "Data Preprocessor",
                                   prog = "Preprocessor",
                                   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type = str, default = './data_preparation/input/datasets/Nipah/Nipah.csv')
    parser.add_argument("--save_path", type = str, default = './data_preparation/output/datasets/Nipah/Nipah_pre.csv')
    args = parser.parse_args()

    data = pd.read_csv(args.input_path)
    columns_to_save = ['PMID','Review_Paper', 'Review']
    preprocessed_df = preprocess_input(data, columns_to_save)
    output_dir = os.path.dirname(args.save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    preprocessed_df.to_csv(args.save_path, index = False)

if __name__ == '__main__':
    main()

