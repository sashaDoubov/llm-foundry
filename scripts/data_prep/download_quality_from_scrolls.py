# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random

import tqdm
from composer.utils import (get_file, maybe_create_object_store_from_uri,
                            parse_uri)
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

data = load_dataset('tau/scrolls', 'quality')
import re

random.seed(0)


def prep_quality_q_last(question, answers, gold, context):
    return {
        'query': f'Text: {context}\nQuestion: {question}\nAnswer:',
        'choices': answers,
        'gold': gold
    }


def prep_quality_q_first(answers, gold, context):
    return {'query': context, 'choices': answers, 'gold': gold}


def extract_question_answers(input_string):
    pattern = r'(?s)^(.*?)\s*\n\s*\((A)\)\s*(.*?)\s*\n\s*\((B)\)\s*(.*?)\s*\n\s*\((C)\)\s*(.*?)\s*\n\s*\((D)\)\s*(.*?)\s*\n\s*(.*)$'
    match = re.match(pattern, input_string)

    if match:
        question = match.group(1).strip()
        answer_1 = match.group(3).strip()
        answer_2 = match.group(5).strip()
        answer_3 = match.group(7).strip()
        answer_4 = match.group(9).strip()
        rest_of_string = match.group(10).strip()

        return question, [answer_1, answer_2, answer_3,
                          answer_4], rest_of_string

    return None


split = 'validation'
base_oci_path = 'oci://mosaicml-internal-datasets/scrolls-icl'

# perform tokenization to print out how many (max) tokens are present
# in the context
tokenize_lengths = True  #False  #True

# optionally upload to remote storage
upload = True

# should the question go first or at the end of the prompt
# Quality by default should have question_last = False
question_last = True

if question_last:
    base_name = 'quality_q_last'
else:
    base_name = 'quality'

if tokenize_lengths:
    tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')

data_length = len(data[split])
for num, out_file in zip([10, 100, 500, len(data[split])], [
        f'/root/{base_name}_small.jsonl', f'/root/{base_name}_medium.jsonl',
        f'/root/{base_name}_500_samples.jsonl', f'/root/{base_name}_full.jsonl'
]):
    if tokenize_lengths:
        max_token_length = 0
    print(data_length)
    if num < data_length:
        sequence = random.sample(list(range(data_length)), num)
    else:
        sequence = range(data_length)

    with open(out_file, 'w', encoding='utf8') as f:

        for iter_num, i in enumerate(tqdm.tqdm(sequence)):

            input_data = data[split]['input'][i]
            question, answers, context = extract_question_answers(input_data)

            out = data[split]['output'][i].strip()
            bool_mask = [out == a for a in answers]

            if not any(bool_mask):
                print(question)
                print(answers)
                print(out)
                raise Exception()

            if question_last:
                row = prep_quality_q_last(question, answers,
                                          bool_mask.index(True), context)
            else:
                row = prep_quality_q_first(answers, bool_mask.index(True),
                                           input_data)

            if tokenize_lengths:
                max_token_length = max(max_token_length,
                                       len(tokenizer.encode(context)))

            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    if upload:
        upload_path = os.path.join(base_oci_path, os.path.basename(out_file))
        print(f'uploading to {upload_path}')
        object_store = maybe_create_object_store_from_uri(upload_path)
        if object_store is not None:
            # remove bucket name and upper part of oci path
            _, _, prefix = parse_uri(upload_path)
            print(prefix)
            object_store.upload_object(prefix, out_file)
    if tokenize_lengths:
        print(f'max token length: {max_token_length}')
