#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import tiktoken
import argparse
import openai
openai.api_key = "Your Key"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
import sys
sys.path.insert(0, './models')
from funs import get_embedding, setup_logger


def main():

    parser = argparse.ArgumentParser(description="Generate embedding",
                                     prog="Embedding Generator",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, default='./data_preparation/output/datasets/Nipah/Nipah_pre_explanations.csv')
    parser.add_argument("--save_path", type=str, default='./data_preparation/output/datasets/Nipah/Nipah_pre_emb.npz')
    parser.add_argument("--log_path", type=str, default='./data_preparation/output/datasets/Nipah/get_embeddings.log')

    args = parser.parse_args()
    save_path = args.save_path
    logger = setup_logger(args.log_path)
    data = pd.read_csv(args.input_path)
    embeddings = []
    ids = []
    for ix, row in data.iterrows():
        try:
            embedding = get_embedding(context = row['Combined'],
                                      model="text-embedding-ada-002",
                                      encoding = "cl100k_base",
                                      max_tokens = 8000)

        except Exception as e:
            logger.error(f"Exception embedding occurred with PMID: {row['PMID']}, error: {e}", exc_info=True)
            embedding = np.array([-1] * 1536)
            time.sleep(60)
        embeddings.append(embedding)
        ids.append(row["PMID"])
        time.sleep(2 if ix % 5 == 0 and ix != 0 else 1)

    embeddings = np.array(embeddings)
    ids = np.array(ids)
    np.savez(save_path, embedding = embeddings, PMID=ids)

if __name__ == '__main__':
    main()






























