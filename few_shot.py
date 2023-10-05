# #!/usr/bin/env python
# ### author: Jingmei Yang: jmyang@bu.edu
#
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import openai
openai.api_key = "Your Key"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
from typing import List, Dict, Tuple
from funs import clean_text, contains_unclear_words
import re

class FewShot:
    """
    The FewShot class is a utility for generating prompts for few-shot learning tasks. It takes a training DataFrame,
    along with several other settings, and can generate a prompt that includes a positive and a negative example
    from the training set, along with their respective questions and answers, the description, the context, and the question.
    """

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
        """
        Constructor for the FewShot class.

        Args:
        train_df : pd.DataFrame
            The DataFrame that contains the training data for few-shot learning.
        system (str): A system message used in gpt-3.5-turbo model and gpt-4 (default vs. customized).
        definition (str): A string that contains descriptions/ definitions of the biomedical terms.
        question (str): A question to be answered.
        cot_prompt (str): Chain of Thought prompt if COT is enabled.
        noncot_prompt (str): Non-CoT prompt if COT is not enabled.
        subquestion_prompt (str): Sub-question prompt if SUB is  enabled.
        explanation_column : str, optional.  The name of the column in the DataFrame that contains explanations. Default is 'Generated_justification'.
        positive_first : bool, optional Whether to present the positive example first in the prompt. Default is True.
        COT (bool, optional): Whether to enable Chain of Thought prompting. Default is False.
        SUB (bool, optional): Whether to enable Sub-question prompting. Default is False.
        """
        if model_type not in ['text-davinci-003', 'gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-4','gpt-3.5-turbo-0301']:
            raise ValueError("model_type must be one of ['text-davinci-003', 'gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-3.5-turbo-0301', 'gpt-4']")
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

    def __str__(self):
        return (f"Initiated FewShot Instance:\n"
                f"Model: {self.model_type}\n"
                f"System Message: {self.system}\n"
                f"Definition: {self.definition}\n"
                f"Chain of Thought: {self.cot}\n"
                f"Sub-questions: {self.sub}\n"
                f"Question: {self.question_prompt}\n"
                f"Explanation Column: {self.explanation_column}\n"
                f"Positive Example First: {self.positive_first}\n"
                f"# Positive Training Examples: {len(self.pos_set)}\n"
                f"# Negative Training Examples: {len(self.neg_set)}\n")


    def apply_train_set_filter(self):
        """
        Divides the training data into positive and negative sets based on the label and the presence of an explanation.

        Returns:
        tuple of pd.DataFrame
            The positive and negative examples as two separate DataFrames.
        """
        self.train_df['Label'] =self.train_df['Review'].apply(lambda x: 1 if 'YES' in x.upper() else 0)

        # if  (self.sub) or (self.cot):
        #     pos_set = self.train_df[(self.train_df[f"{self.explanation_column}_TF"] == 1) & (self.train_df['Label'] == 1)].copy()
        #     neg_set = self.train_df[(self.train_df[f"{self.explanation_column}_TF"] == 0) & (self.train_df['Label'] == 0)].copy()


        if (self.sub) or (self.cot):
            print("Filtering FP & FN")
            explanation_column_tf = f"{self.explanation_column}_TF"

            # Filter pos_set
            # pos_set = self.train_df[(self.train_df[explanation_column_tf] == 1) &
            #                         (self.train_df['Label'] == 1) &
            #                         (~self.train_df[self.explanation_column].apply(
            #                             contains_unclear_words))].copy()
            # neg_set = self.train_df[(self.train_df[explanation_column_tf] == 0) &
            #                         (self.train_df['Label'] == 0) &
            #                         (~self.train_df[self.explanation_column].apply(
            #                             contains_unclear_words))].copy()
            pos_set = self.train_df[(self.train_df[explanation_column_tf] == 1) &
                                    (self.train_df['Label'] == 1)].copy()


            # Filter neg_set
            neg_set = self.train_df[(self.train_df[explanation_column_tf] == 0) &
                                    (self.train_df['Label'] == 0)].copy()
            print(f"pos_set:{pos_set.shape}")
            print(f"neg_set:{neg_set.shape}")





        else:
            pos_set = self.train_df[self.train_df['Label'] == 1].copy()
            neg_set = self.train_df[self.train_df['Label'] == 0].copy()
        return pos_set, neg_set

    def format_answer(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        Formats the paper and answer/explanation from the input DataFrame.

        Args:
        df : pd.DataFrame
            A DataFrame containing a paper, explanation, and answer.

        Returns:
        tuple of str
            A tuple containing the formatted paper and answer/explanation.
        """
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
        """
        Samples one example from the set and formats the answer.

        Args:
        set_df : pd.DataFrame
            The DataFrame from which an example is to be sampled.

        Returns:
        tuple of str
            The formatted paper and answer/explanation.
        """
        df_sample = set_df.sample(n=1).reset_index(drop=True)
        return self.format_answer(df_sample)

    def get_prompt(self, context: str):
        """
        Generates a prompt for the few-shot learning task.

        Args:
        context : str
            The context to include in the prompt.

        Returns:
        str or list
            The generated prompt based on the specified model type. A string for 'text-davinci-003' and a list for 'gpt-3.5-turbo' and 'gpt-4'.
        """

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

        # Generate the prompt based on the specified model type
        if self.model_type == 'text-davinci-003':
            return '\n'.join(p['content'] for p in prompts)
        if self.model_type in ['gpt-3.5-turbo','gpt-3.5-turbo-16k','gpt-3.5-turbo-0301', 'gpt-4']:
            return [{"role": "system", "content": f"{self.system}"}] + prompts

