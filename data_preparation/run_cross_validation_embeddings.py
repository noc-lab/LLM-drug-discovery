#!/usr/bin/env python
#author: Jingmei Yang: jmyang@bu.edu
import os
import argparse
import numpy as np
from cross_validation import CrossValidation
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def main():

    parser=argparse.ArgumentParser(description="Cross validation for embeddings",
                                   prog = "Cross Validator for embeddings",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_fold", type=int, default=5)
    parser.add_argument("--embedded_path", type=str, default='./data_preparation/output/datasets/Nipah/Nipah_pre_emb.npz')
    parser.add_argument("--save_dir", type=str, default='./data_preparation/output/datasets/Nipah/cross_validation_datasets')

    args = parser.parse_args()
    NFOLD = args.num_fold
    SAVE_CROSS_DIR = os.path.join(args.save_dir, 'general')

    if not os.path.exists(SAVE_CROSS_DIR):
        os.makedirs(SAVE_CROSS_DIR)

    embedding_data = np.load(args.embedded_path)
    cv = CrossValidation(n_splits=NFOLD, save_path=SAVE_CROSS_DIR)
    splits = cv.load_splits(save_split_file = 'cross_validation_splits.pkl')

    for i, (train_index, test_index) in enumerate(splits):
        np.savez(os.path.join(SAVE_CROSS_DIR, f"embed_train_{i}.npz"),
                 embedding =  embedding_data["embedding"][train_index],
                 PMID = embedding_data["PMID"][train_index])

        np.savez(os.path.join(SAVE_CROSS_DIR, f"embed_test_{i}.npz"),
                 embedding =  embedding_data["embedding"][test_index],
                 PMID = embedding_data["PMID"][test_index])

if __name__ == '__main__':
    main()





