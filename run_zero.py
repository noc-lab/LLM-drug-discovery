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
    parser=argparse.ArgumentParser(description="Running Zero-Shot settings",
                                   prog = "Zero-Shot",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo-16k",
                        help=" gpt-3.5-turbo vs. gpt-3.5-turbo-16k vs. gpt-4 vs. text-davinci-003 ")
    parser.add_argument("--temperature", type=float, default=0,
                        help="0: model generates sentences in a more deterministic way; 1: model generate sentences in a more diverse way")
    parser.add_argument("--system_path", type=str, default="./models/input/system.txt",
                        help="system message used to define the role of GPT system (gpt-3.5-turbo and more advanced models): default message (system_default.txt) or customized message (system.txt)")
    parser.add_argument("--question_path", type=str, default='./models/input/q2.txt',
                        help = "path to the question file: q1.txt (identify, experimental data) vs. q2.txt (identify, a wet-lab approach) vs. q3.txt (mention, experimental data) vs. q4.txt (identify, a experimental approach)")
    parser.add_argument("--definition_path", type=str, default='./models/input/definitions_wodrug_wetlab_wovirus.txt',
                        help="path to definitions of biomedical terms. binding, drug target, wet-lab")
    parser.add_argument("--cot_path", type=str, default='./models/input/cot_prompt.txt', help="path to a prompt to trigger chain of thought prompting")
    parser.add_argument("--noncot_path", type=str, default='./models/input/noncot_prompt.txt', help="path to a prompt to guide the model to answer a question")
    parser.add_argument("--sub_path", type=str, default='./models/input/Nipah_subquestion_prompt.txt', help="path to a prompt to trigger sub-questions prompting")
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds used in cross validation")
    parser.add_argument("--save_dir", type=str, default='./data_preparation/output/datasets/Nipah/cross_validation_datasets/general',
                        help="directory to the n folds cross-validation sets")
    parser.add_argument("--output_folder", type=str, default='./models/output/Nipah/cross_validation_output', help="directory to save the results")
    parser.add_argument("--cot", type=int, default=1, help = "indicator for chain of thought prompting")
    parser.add_argument("--sub", type=int, default=1, help = "indicator for sub-question prompting")
    parser.add_argument("--answer_col", type=str, default='temp_test', help="column name to save the answer")
    parser.add_argument("--log_path", type=str, default='./models/output/Q2_GeneralDefinitions/Nipah/cross_validation_output/zero_shot.log', help="logging file name")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logger = setup_logger(args.log_path)

    for fold in range(args.num_fold):

        train_df, test_df = load_train_test_data(args.save_dir, fold)

        # Initialize the ZeroShot model
        zeroshot = ZeroShot(model_type = args.gpt_model,
                            system = read_file(args.system_path),
                            definition= read_file(args.definition_path),
                            question = read_file(args.question_path),
                            cot_prompt = read_file(args.cot_path),
                            noncot_prompt=read_file(args.noncot_path),
                            subquestion_prompt=read_file(args.sub_path),
                            COT = bool(args.cot),
                            SUB=bool(args.sub))

        # Initialize the GPT model
        gpt_model = GPTModel(model=zeroshot,
                             model_type=args.gpt_model,
                             temperature=args.temperature,
                             logger=logger,
                             file_path=args.save_dir)
        csv_writer, csv_file = gpt_model.setup_output_directory(args.output_folder, args.answer_col, fold)

        logger.info(f"\nZero Shot:  {args.answer_col} Fold: {fold}.")
        logger.info(f"{zeroshot}")

        # Get test output and write results to CSV
        gpt_model.get_test_output(test_df, csv_writer, args.output_folder, args.answer_col, fold)
        # Close the CSV file after writing
        csv_file.close()
    print(f"Zero Shot Completed:  {args.answer_col}")
if __name__ == '__main__':
    main()



































