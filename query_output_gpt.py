#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import os, time,csv
import numpy as np
import pandas as pd
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
import openai
openai.api_key = "Your Key"
from funs import get_output_file,clean_text, get_answer
from few_shot_similarity import SimilarShot
from zero_shot import ZeroShot
from few_shot import FewShot

def load_train_test_data(file_path, fold):
    """
    Load train and test data for a given fold.

    Args:
        fold (int): The fold number for the train/test split.

    Returns:
        tuple: A tuple of two pandas.DataFrame objects. The first is the training data for the given fold,
        and the second is the testing data for the given fold.
    """
    train_df = pd.read_csv(f'{file_path}/train_{fold}.csv')
    test_df = pd.read_csv(f'{file_path}/test_{fold}.csv')
    return train_df, test_df

class GPTModel:
    def __init__(self, model, model_type, temperature, logger, file_path):
        """
        Initialize a GPTModel object.

        Args:
            model: The model used for prompt generation.
            model_type (str): The type of model used for answer generation.
            temperature (float): The temperature for GPT answer generation.
            logger (Logger): The logger used for error logging.
            file_path (str): The directory where the train/test files are located.
        """
        self.model = model
        self.model_type = model_type
        self.temperature = temperature
        self.logger = logger
        self.file_path = file_path


    def setup_output_directory(self, output_folder, answer_col, fold):
        """
        Set up the output directory for storing CSV results and return CSV writer.

        Args:
            output_folder (str): The directory where the output CSV file is to be stored.
            answer_col (str): The column name for the predicted answers.
            fold (int): The fold number for the train/test split.

        Returns:
            tuple: A tuple containing the csv writer and csv file objects.
        """
        csv_file = open(get_output_file(os.path.join(output_folder, answer_col), f"{answer_col}_{fold}.csv"), 'w')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['PMID', 'Review_Paper', 'Review', f"{answer_col}_{fold}", 'Combined'])
        return csv_writer, csv_file

    def load_test_embeddings(self, fold):
        """
        Load the test embeddings for a given fold.

        Args:
            fold (int): The fold number for the train/test split.

        Returns:
            np.ndarray: The test embeddings for the given fold.
        """
        test_embeddings = np.load(os.path.join(self.file_path, f'embed_test_{fold}.npz'))["embedding"]
        return test_embeddings

    def get_test_output(self, test_df, csv_writer, output_folder, answer_col, fold):
        """
        Generate the test output for a given test DataFrame.

        Args:
            test_df (pandas.DataFrame): The DataFrame containing the test data.
            csv_writer (object): The CSV writer object used for outputting predictions.
            output_folder (str): The directory where the output files are to be stored.
            answer_col (str): The column name for the predicted answers.
            fold (int): The fold number for the train/test split.
        """
        # If the model is an instance of SimilarShot, load the test embeddings
        if isinstance(self.model, SimilarShot):
            test_embeddings = self.load_test_embeddings(fold)

        for ix, row in test_df.iterrows():
            print(f"Starting the abstract #{ix}")

            # Generate the prompt based on the model type
            if isinstance(self.model, SimilarShot):
                prompt = self.model.get_prompt(context=row['Combined'], test_embedding=test_embeddings[ix])
            else:
                prompt = self.model.get_prompt(context=row['Combined'])
            # Save the prompt to a text file
            print(f"#"*30)
            # print(f"prompt{prompt}")
            prompt_dir = os.path.join(output_folder, answer_col, f"{answer_col}_{fold}")
            os.makedirs(prompt_dir, exist_ok=True)
            with open(os.path.join(prompt_dir, f"{row['PMID']}.txt"), 'w') as file:
                file.write(str(prompt))

            # Generate the prediction using the GPT model
            try:
                answer = clean_text(get_answer(model=self.model_type, prompt=prompt, TMP=self.temperature))
            except Exception as e:
                self.logger.error(f"Exception occurred with PMID: {row['PMID']}, error: {e}", exc_info=True)
                answer = 'ERROR'
                time.sleep(60)
            print(f"GPT-3 Answer: {answer}")
            print(f"True Review: {row['Review']}")

            # Write the prediction to the CSV file
            csv_writer.writerow([row['PMID'], row['Review_Paper'], row['Review'], answer, row['Combined']])
            #
            # # Add sleep time to avoid overloading the API
            time.sleep(2 if ix % 5 == 0 and ix != 0 else 1)
