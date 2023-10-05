#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu
import os, argparse
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from funs import read_file, setup_logger
from few_shot import FewShot
from query_output_gpt import GPTModel, load_train_test_data
import openai
openai.api_key = "Your Key"
RANDOM_STATE = 123
np.random.seed(RANDOM_STATE)


def parse_args():
    parser=argparse.ArgumentParser(description="Running a few shot setting",
                                   prog = "Few-Shot",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sars", type=int, default=0, help = "if 1, select from the COVID training set for Nipah testing papers as training examples; 0, otherwise.")
    parser.add_argument("--covid_path", type=str, default='./data_preparation/output/datasets/COVID/cross_validation_datasets/general', help="path to the covid training sets")
    parser.add_argument("--question_path", type=str, default='./models/input/q2.txt',
                        help = "path to the question file: q1.txt (identify, experimental data) vs. q2.txt (identify, a wet-lab approach) vs. q3.txt (mention, experimental data) vs. q4.txt (identify, a experimental approach)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="0: model generates sentences in a more deterministic way; 1: model generate sentences in a more diverse way")
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo-16k",
                        help=" gpt-3.5-turbo vs. gpt-3.5-turbo-16k vs. gpt-4 vs. text-davinci-003 ")
    parser.add_argument("--cot_path", type=str, default='./models/input/cot_prompt.txt',  help="path to a prompt to trigger chain of thought prompting")
    parser.add_argument("--noncot_path", type=str, default='./models/input/noncot_prompt.txt',  help="path to a prompt to guide the model to answer a question")
    parser.add_argument("--sub_path", type=str, default='./models/input/Nipah_subquestion_prompt.txt', help="path to a prompt to trigger sub-questions prompting")
    parser.add_argument("--num_fold", type=int, default=5, help="number of folds used in cross validation")
    parser.add_argument("--definition_path", type=str, default='./models/input/definitions_wodrug_wetlab_wovirus.txt',
                        help="path to definitions of biomedical terms. binding, drug target, wet-lab")

    parser.add_argument("--save_dir", type=str, default='./data_preparation/output/datasets/Nipah/cross_validation_datasets/general',
                        help="directory to the n folds cross-validation sets")
    parser.add_argument("--output_folder", type=str, default='./models/output/Nipah/cross_validation_output', help="directory to save the GPT outputs")

    parser.add_argument("--system_path", type=str, default="./models/input/system.txt",
                        help="system message used to define the role of GPT system (gpt-3.5-turbo and more advanced models): default message (system_default.txt) or customized message (system.txt)")
    parser.add_argument("--sub", type=int, default=1, help = "indicator for sub-question prompting")
    parser.add_argument("--cot", type=int, default=1, help = "indicator for CoT prompting")
    parser.add_argument("--explanation_col", type=str, default='Generated_justification', help="type of generated explanation used in CoT: Generated_zeroshot_cot vs. Generated_justification")
    parser.add_argument("--positive_first", type=int, default=0, help = "if 1, use a positive example as the first example in the prompt; 0, otherwise.")
    parser.add_argument("--answer_col", type=str, default='Nipah_Q2_Turbo_Customized_FewCoT_GeneratedJustification_NegFirst', help="column name to save the answer")
    parser.add_argument("--log_path", type=str, default='./models/output/Nipah/cross_validation_output/few_shot.log', help="file name for logging")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logger = setup_logger(args.log_path)

    for fold in range(args.num_fold):

        # Load train and test data
        train_df, test_df = load_train_test_data(args.save_dir, fold)

        # Use COVID papers as training examples if 'sars'  is True
        if bool(args.sars):
            train_df = pd.read_csv(f'{args.covid_path}/train_{fold}.csv')

        # Initialize the FewShot model
        fewshot = FewShot(train_df=train_df,
                          model_type=args.gpt_model,
                          system=read_file(args.system_path),
                          definition=read_file(args.definition_path),
                          question=read_file(args.question_path),
                          cot_prompt=read_file(args.cot_path),
                          noncot_prompt=read_file(args.noncot_path),
                          subquestion_prompt=read_file(args.sub_path),
                          positive_first=bool(args.positive_first),
                          explanation_column=args.explanation_col,
                          SUB=bool(args.sub),
                          COT=bool(args.cot))



        # Initialize the GPT model
        gpt_model = GPTModel(model=fewshot,
                             model_type=args.gpt_model,
                             temperature=args.temperature,
                             logger=logger,
                             file_path=args.save_dir)

        # Setup output directory
        csv_writer, csv_file = gpt_model.setup_output_directory(args.output_folder, args.answer_col, fold)

        # Log the model type and fold number
        logger.info(f"\nFew Shot:  {args.answer_col} \nFold: {fold}\n")
        logger.info(f"{gpt_model.model}")
        logger.info(f"{args.explanation_col}")

        # Get test output and write results to CSV
        gpt_model.get_test_output(test_df, csv_writer, args.output_folder, args.answer_col, fold)

        csv_file.close()

    print(f"Few Shot Completed:  {args.answer_col}")


if __name__ == '__main__':
    main()














