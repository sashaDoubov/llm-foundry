# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import tqdm
from composer.utils import (get_file, maybe_create_object_store_from_uri,
                            parse_uri)
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

data = load_dataset('tau/scrolls', 'quality')
import re


def prep_quality(question, answers, gold, context):
    return {
        'query': f'Text: {context}\nQuestion: {question}\nAnswer:',
        'choices': answers,
        'gold': gold
    }


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


# Example usage

split = 'validation'
base_oci_path = 'oci://mosaicml-internal-datasets/scrolls-icl'
tokenize_lengths = False  #True
upload = True

if tokenize_lengths:
    tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')
for num, out_file in zip([10, 100, len(data[split])], [
        '/root/quality_small.jsonl', '/root/quality_medium.jsonl',
        '/root/quality_full.jsonl'
]):
    if tokenize_lengths:
        max_token_length = 0
    with open(out_file, 'w', encoding='utf8') as f:
        for i in tqdm.tqdm(range(num)):

            question, answers, context = extract_question_answers(
                data[split]['input'][i])

            out = data[split]['output'][i].strip()
            bool_mask = [out == a for a in answers]

            if not any(bool_mask):
                print(question)
                print(answers)
                print(out)
                raise Exception()

            row = prep_quality(question, answers, bool_mask.index(True),
                               context)

            if tokenize_lengths:
                max_token_length = max(max_token_length,
                                       len(tokenizer.encode(context)))

            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    if upload:
        upload_path = os.path.join(base_oci_path, os.path.basename(out_file))
        object_store = maybe_create_object_store_from_uri(upload_path)
        if object_store is not None:
            # remove bucket name and upper part of oci path
            _, _, prefix = parse_uri(upload_path)
            print(prefix)
            object_store.upload_object(prefix, out_file)
    if tokenize_lengths:
        print(max_token_length)
