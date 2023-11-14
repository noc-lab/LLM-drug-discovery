#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import os, time, csv
import numpy as np
import pandas as pd
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
import openai
openai.api_key = "Your Key"
from funs import get_output_file,clean_text, get_answer
from few_shot_similarity import SimilarShot

def load_train_test_data(file_path, fold):
    train_df = pd.read_csv(f'{file_path}/train_{fold}.csv')
    test_df = pd.read_csv(f'{file_path}/test_{fold}.csv')
    return train_df, test_df

class GPTModel:
    def __init__(self, model, model_type, temperature, logger, file_path):
        self.model = model
        self.model_type = model_type
        self.temperature = temperature
        self.logger = logger
        self.file_path = file_path

    def setup_output_directory(self, output_folder, answer_col, fold):
        csv_file = open(get_output_file(os.path.join(output_folder, answer_col), f"{answer_col}_{fold}.csv"), 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['PMID', 'Review_Paper', 'Review', f"{answer_col}_{fold}", 'Combined'])
        return csv_writer, csv_file

    def load_test_embeddings(self, fold):
        test_embeddings = np.load(os.path.join(self.file_path, f'embed_test_{fold}.npz'))["embedding"]
        return test_embeddings

    def get_test_output(self, test_df, csv_writer, output_folder, answer_col, fold):
        if isinstance(self.model, SimilarShot):
            test_embeddings = self.load_test_embeddings(fold)

        for ix, row in test_df.iterrows():
            if isinstance(self.model, SimilarShot):
                prompt = self.model.get_prompt(context=row['Combined'], test_embedding=test_embeddings[ix])
            else:
                prompt = self.model.get_prompt(context=row['Combined'])

            prompt_dir = os.path.join(output_folder, answer_col, f"{answer_col}_{fold}")
            os.makedirs(prompt_dir, exist_ok=True)

            with open(os.path.join(prompt_dir, f"{row['PMID']}.txt"), 'w') as file:
                file.write(str(prompt))

            try:
                answer = clean_text(get_answer(model=self.model_type, prompt=prompt, TMP=self.temperature))

            except Exception as e:
                self.logger.error(f"Exception occurred with PMID: {row['PMID']}, error: {e}", exc_info=True)
                answer = 'ERROR'
                time.sleep(60)

            csv_writer.writerow([row['PMID'], row['Review_Paper'], row['Review'], answer, row['Combined']])
            time.sleep(2 if ix % 5 == 0 and ix != 0 else 1)
