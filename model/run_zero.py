#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import openai
from zero_shot import ZeroShot
from funs import read_file, setup_logger
from query_output_gpt import GPTModel, load_train_test_data
openai.api_key = "Your Key"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)

def parse_args():
    parser=argparse.ArgumentParser(description="Running Zero-Shot",
                                   prog = "Zero-Shot",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--system_path", type=str, default="./model/input/system.txt")
    parser.add_argument("--question_path", type=str, default='./model/input/Q2.txt')
    parser.add_argument("--definition_path", type=str, default='./model/input/definitions.txt')
    parser.add_argument("--cot_path", type=str, default='./model/input/cot_prompt.txt')
    parser.add_argument("--noncot_path", type=str, default='./model/input/noncot_prompt.txt')
    parser.add_argument("--sub_path", type=str, default='./model/input/subquestion_prompt.txt')
    parser.add_argument("--num_fold", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default='./data_preparation/output/datasets/Nipah/cross_validation_datasets/general')
    parser.add_argument("--output_folder", type=str, default='./model/output/Nipah/cross_validation_output')
    parser.add_argument("--cot", type=int, default=1)
    parser.add_argument("--sub", type=int, default=0)
    parser.add_argument("--answer_col", type=str, default='Nipah_Q2_ZeroCoT')
    parser.add_argument("--log_path", type=str, default='./model/output/Nipah/cross_validation_output/zero_shot.log')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logger = setup_logger(args.log_path)

    for fold in range(args.num_fold):
        train_df, test_df = load_train_test_data(args.save_dir, fold)

        zeroshot = ZeroShot(model_type = args.gpt_model,
                            system = read_file(args.system_path),
                            definition= read_file(args.definition_path),
                            question = read_file(args.question_path),
                            cot_prompt = read_file(args.cot_path),
                            noncot_prompt = read_file(args.noncot_path),
                            subquestion_prompt = read_file(args.sub_path),
                            COT = bool(args.cot),
                            SUB = bool(args.sub))

        gpt_model = GPTModel(model = zeroshot,
                             model_type = args.gpt_model,
                             temperature = args.temperature,
                             logger = logger,
                             file_path = args.save_dir)
        csv_writer, csv_file = gpt_model.setup_output_directory(args.output_folder, args.answer_col, fold)

        logger.info(f"\nZero Shot:  {args.answer_col} Fold: {fold}.")
        logger.info(f"\nOutput Directory: {args.output_folder}")

        gpt_model.get_test_output(test_df, csv_writer, args.output_folder, args.answer_col, fold)
        csv_file.close()

if __name__ == '__main__':
    main()



































