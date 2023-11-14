#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import openai
from typing import Tuple
openai.api_key = "Your Key"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
from funs import clean_text


class SimilarShot:
    def __init__(self, train_df: pd.DataFrame,
                 embedd_train_file: str,
                 model_type: str,
                 system: str,
                 definition: str,
                 question: str,
                 cot_prompt: str,
                 noncot_prompt: str,
                 subquestion_prompt: str,
                 explanation_column='Generated_justification',
                 top_n_similar=1,
                 COT: bool = False,
                 SUB: bool = False) -> None:

        if not all(isinstance(i, str) for i in
                   [system, definition, question, cot_prompt, noncot_prompt, subquestion_prompt, explanation_column, embedd_train_file]):
            raise TypeError(
                "system, description, question, cot_prompt, noncot_prompt, subquestion_prompt, embedd_train_file,  and explanation_column must be string.")
        if not isinstance(COT, bool):
            raise TypeError("COT must be bool.")
        if not isinstance(SUB, bool):
            raise TypeError("SUB must be bool.")
        if not isinstance(top_n_similar, int):
            raise TypeError("top_n_similar must be int.")

        self.cot_prompt = cot_prompt
        self.cot = COT
        self.sub = SUB
        self.definition = definition
        self.system = system
        self.model_type = model_type
        self.explanation_column = explanation_column
        self.top_n_similar = top_n_similar
        self.apply_train_set_filter(train_df, embedd_train_file)

        if top_n_similar > len(self.train_df):
            raise ValueError(
                f"top_n_similar ({top_n_similar}) is greater than the number of training examples ({len(self.train_df)})")
        if  self.sub:
            self.question_prompt = f"Primary question: {question}\n{subquestion_prompt}"
        elif self.cot:
            self.question_prompt = f"Question: {question}\n{cot_prompt}"
        else:
            self.question_prompt = f"Question: {question}\n{noncot_prompt}"

    def apply_train_set_filter(self, train_df: pd.DataFrame, embedd_train_file: str) -> None:

        data = np.load(embedd_train_file)

        if  (self.sub) or (self.cot):
            train_df['Label'] = train_df['Review'].apply(lambda x: 1 if  'YES' in x.upper() else 0)
            mask = (train_df[f'{self.explanation_column}_TF'] == train_df['Label'])
            self.train_df = train_df[mask]
            self.embeddings = data["embedding"][mask]
            self.ids = data["PMID"][mask]

        else:
            self.train_df = train_df
            self.embeddings = data["embedding"]
            self.ids = data["PMID"]

    def calculate_cosine_similarities(self, embeddings: np.ndarray, test_embedding: np.ndarray) -> np.ndarray:

        return np.dot(embeddings, test_embedding) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(test_embedding))

    def find_similar_examples(self, test_embedding: np.ndarray) -> pd.DataFrame:
        cosine_similarities = self.calculate_cosine_similarities(self.embeddings, test_embedding)
        cosine_df = pd.DataFrame({'PMID': self.ids, 'cosine_similarity': cosine_similarities})
        merged_df = pd.merge(self.train_df, cosine_df, on='PMID')
        sorted_df = merged_df.sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
        similar_examples = sorted_df.head(self.top_n_similar)
        return similar_examples

    def format_answer(self, df: pd.DataFrame, ix: int) -> Tuple[str, str]:
        if df.empty:
            raise ValueError("DataFrame is empty.")

        paper, explanation, review = df.iloc[ix][['Combined', self.explanation_column, 'Review']]
        explanation = clean_text(explanation)
        if self.sub:
            return (paper, explanation)
        elif self.cot:
            return (paper, f"{explanation}")
        else:
            return (paper, f"{review}.")

    def get_prompt(self, context: str, test_embedding: np.ndarray):
        similar_top2 = self.find_similar_examples(test_embedding)
        if similar_top2.empty:
            raise ValueError("No similar examples found.")

        prompts = []
        for ix in range(self.top_n_similar):
            similar_paper, similar_a = self.format_answer(similar_top2, ix)
            if ix == 0:
                prompts.append({"role": "user", "content": f"{self.definition}\n\n{similar_paper}\n\n{self.question_prompt}"})
            else:
                prompts.append({"role": "user", "content": f"{similar_paper}\n\n{self.question_prompt}"})

            prompts.append({"role": "assistant", "content": f"{similar_a}"})

        prompts.append({"role": "user", "content": f"{context}\n\n{self.question_prompt}"})
        if self.model_type == 'text-davinci-003':
            return '\n'.join([p['content'] for p in prompts])
        if self.model_type in ['gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-3.5-turbo-0301', 'gpt-4']:
            return [{"role": "system", "content": f"{self.system}"}] + prompts