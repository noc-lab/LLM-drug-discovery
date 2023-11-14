#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time
import argparse
import openai
from justification_generation import JustificationGenerator
import sys
sys.path.insert(0, './models')
from funs import get_answer, read_file, setup_logger, clean_text
from zero_shot import ZeroShot
openai.api_key = "Your Key"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)



def parse_args():
    parser=argparse.ArgumentParser(description="Explanation Generator",
                                   prog = "Explanation Generator",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--system_path", type=str, default="./models/input/system.txt")
    parser.add_argument("--question_path", type=str, default='./models/input/Q2.txt')
    parser.add_argument("--definition_path", type=str, default='./models/input/definitions.txt')
    parser.add_argument("--cot_path", type=str, default='./models/input/cot_prompt.txt')
    parser.add_argument("--noncot_path", type=str, default='./models/input/noncot_prompt.txt')
    parser.add_argument("--sub_path", type=str, default='./models/input/subquestion_prompt.txt')
    parser.add_argument("--justification_path", type=str, default='./data_preparation/input/justification_prompt.txt')
    parser.add_argument("--input_path", type=str, default='./data_preparation/output/datasets/Nipah/Nipah_pre.csv')
    parser.add_argument("--output_path", type=str, default='./data_preparation/output/datasets/Nipah/Nipah_pre_explanations.csv')
    parser.add_argument("--save_col", type=str, default='Generated_justification')
    parser.add_argument("--sub", type=int, default=0)
    parser.add_argument("--cot", type=int, default=1)
    parser.add_argument("--log_path", type=str, default='./data_preparation/output/datasets/Nipah/generate_explanation.log')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cot = bool(args.cot)
    logger = setup_logger(args.log_path)

    if cot:
        explanation_generator =   ZeroShot(model_type = args.gpt_model,
                            system = read_file(args.system_path),
                            definition= read_file(args.definition_path),
                            question = read_file(args.question_path),
                            cot_prompt = read_file(args.cot_path),
                            noncot_prompt=read_file(args.noncot_path),
                            subquestion_prompt=read_file(args.sub_path),
                            COT = bool(args.cot),
                            SUB=bool(args.sub))
    else:
        explanation_generator = JustificationGenerator(model_type=args.gpt_model, system=read_file(args.system_path),
                                                       definition=read_file(args.definition_path), question=read_file(args.question_path),
                                                       justification_prompt = read_file(args.justification_path))

    data = pd.read_csv(args.input_path)
    generated_explanations = []

    for ix, row in data.iterrows():
        if cot:
            prompt = explanation_generator.get_prompt(context=row['Combined'])
        else:
            prompt = explanation_generator.get_prompt(context=row['Combined'],
                                                      review=row['Review'])

        prompt_dir = os.path.join(os.path.dirname(args.output_path), args.save_col)
        os.makedirs(prompt_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, f"{row['PMID']}.txt"), 'w') as file:
            file.write(str(prompt))
        try:
            answer = clean_text(get_answer(model=args.gpt_model, prompt=prompt, TMP=args.temperature))

        except Exception as e:
            logger.error(f"Exception occurred with PMID: {row['PMID']}, error: {e}", exc_info=True)
            answer = 'ERROR'
            time.sleep(60)

        generated_explanations.append(answer)

    data[args.save_col] = generated_explanations
    data.to_csv(args.output_path, index=False)
    logger.info(f"Explanation Generation for {args.save_col} Completed.")

if __name__ == '__main__':
    main()






























