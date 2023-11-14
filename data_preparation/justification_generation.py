#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

class JustificationGenerator:
    def __init__(self, model_type: str, system: str,
                 definition: str, question: str,
                 justification_prompt: str):

        if not all(isinstance(i, str) for i in [system, definition, question, justification_prompt]):
            raise TypeError("system, definition, question, and justification_prompt must be string.")

        self.definition = definition
        self.question = question
        self.system = system
        self.model_type = model_type
        self.justification_prompt = justification_prompt

    def get_prompt(self, context: str, review: str):
        context_prompt = f"{self.definition}\n{context}\nQuestion: {self.question}\nAnswer: {review}.\n{self.justification_prompt} "

        if self.model_type == 'text-davinci-003':
            return context_prompt

        if self.model_type in ['gpt-3.5-turbo','gpt-3.5-turbo-16k', 'gpt-4']:
            return [
                {"role": "system", "content": f"{self.system}"},
                {"role": "user", "content": f"{context_prompt}"}
            ]






