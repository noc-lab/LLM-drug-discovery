#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import argparse
import pandas as pd
import numpy as np

# Set a random state for reproducibility
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def preprocess_input(df, columns_to_save):
    """
    Preprocess the input DataFrame by cleaning and combining certain columns.

    Parameters:
    df: pandas DataFrame, the input data to be preprocessed.
    columns_to_save: list of str, the column names to be saved in the output.
    """
    df['Keyword'] = df['Keyword'].replace('Missing', None)
    df['Review'] = np.where(df['Review'], 'Yes', 'No')

    # Combine 'Title', 'Keyword', and 'Abstract' into a new 'Combined' column
    df['Combined'] = df.apply(lambda row: ('Paper:\nTitle: ' + row['Title'] + (
        '\n' + row['Keyword'] if row['Keyword'] is not None else '') + '\n' + 'Abstract: ' + row['Abstract']), axis=1)

    columns_to_save.append("Combined")

    return df[columns_to_save]


def main():
    """
    Load the input data, preprocess it, and save the output.
    """
    parser=argparse.ArgumentParser(description="Data Preprocessor",
                                   prog = "Preprocessor",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--input_path",type=str, default='./data_preparation/input/datasets/Nipah/Nipah.csv', help = "Path to the input dataset")
    parser.add_argument("--save_path", type=str, default='./data_preparation/output/datasets/Nipah/Nipah_pre.csv', help="Path to save the preprocessed dataset")
    args = parser.parse_args()

    # Read the input dataset
    data = pd.read_csv(args.input_path)
    print(f"{data.shape[0]} rows found in the input dataset.")

    # Preprocess the input dataset
    columns_to_save = ['PMID','Review_Paper', 'Review']
    preprocessed_df = preprocess_input(data, columns_to_save)

    # Check if the output directory exists, if not, create it
    output_dir = os.path.dirname(args.save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the preprocessed dataset to the specified path
    preprocessed_df.to_csv(args.save_path, index = False)
    print('Concatenation completed.')

if __name__ == '__main__':
    main()

