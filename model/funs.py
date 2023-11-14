import os
import numpy as np
import random, time
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)
import openai
openai.api_key = "Your Key"
import logging
import re


def clean_text(text: str) -> str:
    cleaned_text = re.sub(r'[\n\t]+| {2,}', ' ', text.strip())
    cleaned_text = cleaned_text.replace('"', '')
    cleaned_text = cleaned_text.replace("'", '')
    return cleaned_text.strip()


def setup_logger(log_file, logger_name=__name__, level=logging.INFO):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    handler = logging.FileHandler(log_file)
    handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def read_file(FILEPATH):
    with open(FILEPATH,'r') as f:
      file_text = f.read().strip()
      f.close()
    return file_text



def get_output_file(output_folder, output_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return os.path.join(output_folder,output_file)


def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.error.RateLimitError, openai.error.Timeout, openai.error.ServiceUnavailableError, ),
):

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)

            except errors as e:
                num_retries += 1

                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                delay *= exponential_base * (1 + jitter * random.random())

                time.sleep(delay)

            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def get_answer(model, prompt, TMP = 0):
    if model  == 'text-davinci-003':
        response = openai.Completion.create(
            model= model,
            prompt = prompt,
            temperature = TMP,
            max_tokens = 1000,
            top_p =1,
            logprobs = 1)
        answer = response['choices'][0]['text'].strip()
        return answer

    else:
        response = openai.ChatCompletion.create(
            model= model,
            messages=prompt,
            temperature=TMP,
            max_tokens=1000,
            top_p=1)
        answer = response['choices'][0]['message']['content'].strip()
        return answer


@retry_with_exponential_backoff
def get_embedding(context, model="text-embedding-ada-002", encoding = "cl100k_base", max_tokens = 8000):
    response = openai.Embedding.create(input = context,
                                        model = model,
                                        encoding = encoding,
                                        max_tokens = max_tokens)
    return response["data"][0]["embedding"]





