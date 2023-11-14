#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

class ZeroShot:
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
        if self.sub:
            self.question_prompt = f"Primary question: {question}\n{subquestion_prompt}"
        elif self.cot:
            self.question_prompt = f"Question: {question}\n\n{cot_prompt}"
        else:
            self.question_prompt = f"Question: {question}\n{noncot_prompt}"

    def get_prompt(self, context: str):

        if self.model_type == 'text-davinci-003':
            return f"{self.definition}\n{context}\n{self.question_prompt}"
        if self.model_type in ['gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-4']:
            return [
                {"role": "system", "content": f"{self.system}"},
                {"role": "user", "content": f"{self.definition}\n\n{context}\n{self.question_prompt}"}]

