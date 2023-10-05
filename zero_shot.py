#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu


class ZeroShot:
    """
    This class is designed for creating ZeroShot instances.
    """

    def __init__(self,
                 model_type: str,
                 system: str,
                 definition: str,
                 question: str,
                 cot_prompt: str,
                 noncot_prompt: str,
                 subquestion_prompt: str,
                 COT: bool = False,
                 SUB: bool = False) -> None:
        """
        Initialize a ZeroShot instance.

        Args:
            model_type (str): 3 types of model: text-davinci-003, gpt-3.5-turbo, gpt-4. For gpt-3.5-turbo and gpt-4, the prompt formats are same.
            system (str): A system message used in gpt-3.5-turbo model and gpt-4 (default vs. customized).
            definition (str): A string that contains descriptions/ definitions of the biomedical terms.
            question (str): A question to be answered.
            cot_prompt (str): Chain of Thought prompt if COT is enabled.
            noncot_prompt (str): Non-CoT prompt if COT is not enabled.
            subquestion_prompt (str): Sub-question prompt if SUB is  enabled.
            COT (bool, optional): Whether to enable Chain of Thought prompting. Default is False.
            SUB (bool, optional): Whether to enable Sub-question prompting. Default is False.
        """
        if model_type not in ['text-davinci-003', 'gpt-3.5-turbo', 'gpt-3.5-turbo-0301','gpt-3.5-turbo-16k', 'gpt-4']:
            raise ValueError("model_type must be one of ['text-davinci-003', 'gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-4']")
        if not all(isinstance(i, str) for i in [system, definition, question, cot_prompt, noncot_prompt, subquestion_prompt]):
            raise TypeError("system, definition, question, cot_prompt, noncot_prompt, and subquestion_prompt  must be string.")
        if not isinstance(COT, bool):
            raise TypeError("COT must be bool.")
        if not isinstance(SUB, bool):
            raise TypeError("SUB must be bool.")

        self.model_type = model_type
        self.system = system
        self.definition = definition
        self.cot = COT
        self.sub = SUB
        if  self.sub:
            self.question_prompt = f"Primary question: {question}\n{subquestion_prompt}"
        elif self.cot:
            self.question_prompt = f"Question: {question}\n\n{cot_prompt}"
        else:
            self.question_prompt = f"Question: {question}\n{noncot_prompt}"
    def __str__(self):
        return f"Initiated ZeroShot Instance:\nModel: {self.model_type}\nSystem Message: {self.system}\nBio Definitions: {self.definition}\nChain of Thought: {self.cot}\nSub-questions: {self.sub}\nQuestion: {self.question_prompt}\n"

    def get_prompt(self, context: str):
        """
        Generate a prompt.

        Args:
            context (str): A string that provides the context for the prompt.

        Returns:
            A string or list that represents the generated prompt.
        """
        if self.model_type == 'text-davinci-003':
            return f"{self.definition}\n{context}\n{self.question_prompt}"
        if self.model_type in ['gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-4']:
            return [
                {"role": "system", "content": f"{self.system}"},
                {"role": "user", "content": f"{self.definition}\n\n{context}\n{self.question_prompt}"}]

