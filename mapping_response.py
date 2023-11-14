#!/usr/bin/env python
### author: Jingmei Yang: jmyang@bu.edu

import pandas as pd
import os, argparse
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from funs import get_output_file,setup_logger,clean_text


def check_patterns(patterns, sentence):
    return any(pattern.search(sentence) for pattern in patterns)

def map_text_to_label_cot_string(text):
    YES_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in [r"\byes\b", r"\bdoes\b(?! not)"]]
    
    NO_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in
                   [r"\bno\b", r"\bdoes not\b", r"\bdoesn't\b", r"\bnot\b"]]
    
    NOT_SURE_PATTERNS = [re.compile(pattern, re.IGNORECASE) for pattern in
                         [r"\binsufficient\b", r"\bunclear\b", r"\binconclusive\b", r"\bpartial\b", r"\bimpossible\b", r"\bindeterminate\b"]]

    LABEL_ERROR = -1
    LABEL_NO = 0
    LABEL_YES = 1
    LABEL_CONTRADICT = 2
    LABEL_NOT_SURE = 3
    LABEL_UNDEFINED = 4

    text.strip().lower() + '.' if not text.endswith('.') else text.strip().lower()
    try:
        text = clean_text(text)
    except Exception as e:
        print(e)
        print(f"text:{text:}")

    sentences = re.split('\. (?=[A-Za-z])|\? |\!', text)

    if text.upper() == "ERROR":
        return LABEL_ERROR

    last_sentence = sentences[-1].lower()
    first_sentence = sentences[0].lower()

    if check_patterns(NOT_SURE_PATTERNS, last_sentence):
        last_sentence_label = LABEL_NOT_SURE
    elif check_patterns(NO_PATTERNS, last_sentence):
        last_sentence_label = LABEL_NO
    elif check_patterns(YES_PATTERNS, last_sentence):
        last_sentence_label = LABEL_YES
    else:
        last_sentence_label = LABEL_UNDEFINED

    if check_patterns(NOT_SURE_PATTERNS, first_sentence):
        first_sentence_label = LABEL_NOT_SURE
    elif check_patterns(NO_PATTERNS, first_sentence):
        first_sentence_label = LABEL_NO
    elif check_patterns(YES_PATTERNS, first_sentence):
        first_sentence_label = LABEL_YES
    else:
        first_sentence_label = LABEL_UNDEFINED

    if first_sentence_label == LABEL_NOT_SURE or last_sentence_label == LABEL_NOT_SURE:
        return LABEL_NOT_SURE
    elif first_sentence_label == LABEL_UNDEFINED and last_sentence_label in [LABEL_YES, LABEL_NO]:
        return last_sentence_label
    elif first_sentence_label == last_sentence_label:
        return first_sentence_label
    elif (first_sentence_label in [LABEL_YES, LABEL_NO] and
          last_sentence_label in [LABEL_YES, LABEL_NO]):
        return LABEL_CONTRADICT
    else:
        return LABEL_UNDEFINED
    
def parse_args():

    parser=argparse.ArgumentParser(description="Response mapping",
                                   prog="Response mapping",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    parser.add_argument("--num_fold", type=int, default=5)
    parser.add_argument("--answer_path",type=str, default="./model/output/Nipah/cross_validation_output/answer_columns.txt")
    parser.add_argument("--input_path",type=str, default='./model/output/Nipah/cross_validation_output')
    parser.add_argument("--output_folder", type=str, default='./model/results/Nipah/cross_validation_output')
    parser.add_argument("--log_path", type=str, default='./model/results/Nipah/cross_validation_output/evaluation.log')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logger = setup_logger(args.log_path)
    NFOLD = args.num_fold

    with open(args.answer_path,'r') as f:
        cols = [line.strip() for line in f.readlines()]
        
    for answer_col in cols:
        SUB_FOLDER = os.path.join(args.input_path, answer_col)

        for fold in range(NFOLD):
            ANS_COL = f"{answer_col}_{fold}"
            ANS_TF = f"{ANS_COL}_TF"

            logger.info(f"Mapping  {ANS_COL}!!!!!!!!!!!!!!!!!!")
            df = pd.read_csv(get_output_file(SUB_FOLDER, f"{ANS_COL}.csv"))
            df['Label'] = df['Review'].apply(lambda x: 1 if 'YES' in x.upper() else 0)
            df[ANS_TF] = df[ANS_COL].apply(map_text_to_label_cot_string)

            df[ ['PMID', 'Review_Paper', 'Review', ANS_COL, 'Label', ANS_TF]].to_csv(get_output_file(SUB_FOLDER, f"{ANS_COL}_ans.csv"), index=False)

if __name__ == '__main__':
    main()
