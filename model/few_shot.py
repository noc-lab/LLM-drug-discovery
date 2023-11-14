# #!/usr/bin/env python
# ### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import openai
openai.api_key = "Your Key"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
from typing import Tuple
from funs import clean_text

class FewShot:

    def __init__(self,
                 train_df: pd.DataFrame,
                 model_type: str,
                 system: str,
                 definition: str,
                 question: str,
                 cot_prompt: str,
                 noncot_prompt: str,
                 subquestion_prompt: str,
                 explanation_column = 'Generated_justification',
                 positive_first  = True,
                 COT: bool = False,
                 SUB: bool = False) -> None:

        if not all(isinstance(i, str) for i in [system, definition, question, cot_prompt, noncot_prompt, subquestion_prompt,explanation_column]):
            raise TypeError("system, definition, question,explanation_column, cot_prompt, noncot_prompt, and subquestion_prompt  must be string.")
        if not isinstance(COT, bool):
            raise TypeError("COT must be bool.")
        if not isinstance(SUB, bool):
            raise TypeError("SUB must be bool.")
        if not isinstance(positive_first, bool):
            raise TypeError("positive_first must be bool.")
        self.train_df = train_df
        self.model_type = model_type
        self.system = system
        self.definition = definition
        self.cot = COT
        self.sub = SUB
        self.positive_first = positive_first
        self.explanation_column = explanation_column

        if  self.sub:
            self.question_prompt = f"Primary question: {question}\n{subquestion_prompt}"
        elif self.cot:
            self.question_prompt = f"Question: {question}\n{cot_prompt}"
        else:
            self.question_prompt = f"Question: {question}\n{noncot_prompt}"

        self.pos_set, self.neg_set = self.apply_train_set_filter()

    def apply_train_set_filter(self):
        self.train_df['Label'] =self.train_df['Review'].apply(lambda x: 1 if 'YES' in x.upper() else 0)

        if (self.sub) or (self.cot):
            explanation_column_tf = f"{self.explanation_column}_TF"
            pos_set = self.train_df[(self.train_df[explanation_column_tf] == 1) &
                                    (self.train_df['Label'] == 1)].copy()

            neg_set = self.train_df[(self.train_df[explanation_column_tf] == 0) &
                                    (self.train_df['Label'] == 0)].copy()
        else:
            pos_set = self.train_df[self.train_df['Label'] == 1].copy()
            neg_set = self.train_df[self.train_df['Label'] == 0].copy()

        return pos_set, neg_set

    def format_answer(self, df: pd.DataFrame) -> Tuple[str, str]:
        if df.empty:
            raise ValueError("DataFrame is empty.")

        paper, explanation, review = df.iloc[0][['Combined', self.explanation_column, 'Review']]
        explanation = clean_text(explanation)

        if self.sub:
            return (paper, explanation)
        elif self.cot:
            return (paper, f"{explanation} Therefore, the final answer is {review}.")
        else:
            return (paper, f"{review}.")

    def get_sample_and_format(self, set_df: pd.DataFrame) -> Tuple[str, str]:

        df_sample = set_df.sample(n=1).reset_index(drop=True)

        return self.format_answer(df_sample)

    def get_prompt(self, context: str):
        pos_paper, pos_a = self.get_sample_and_format(self.pos_set)
        neg_paper, neg_a = self.get_sample_and_format(self.neg_set)

        samples = {'positive': (pos_paper, pos_a), 'negative': (neg_paper, neg_a)}
        first_p, first_a = samples['positive' if self.positive_first else 'negative']
        second_p, second_a = samples['negative' if self.positive_first else 'positive']

        prompts = [
            {"role": "user", "content": f"{self.definition}\n\n{first_p}\n\n{self.question_prompt}"},
            {"role": "assistant", "content": f"{first_a}"},
            {"role": "user", "content": f"{second_p}\n\n{self.question_prompt}"},
            {"role": "assistant", "content": f"{second_a}"},
            {"role": "user", "content": f"{context}\n\n{self.question_prompt}"}
        ]
        if self.model_type == 'text-davinci-003':
            return '\n'.join(p['content'] for p in prompts)
        if self.model_type in ['gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-3.5-turbo-0301', 'gpt-4']:
            return [{"role": "system", "content": f"{self.system}"}] + prompts

