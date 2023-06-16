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

dataset_name = 'qasper'
data = load_dataset('tau/scrolls', dataset_name)

random.seed(0)


def prep_narrative(context, answer):
    return {'context': context, 'answer': answer, 'aliases': [answer]}


# Example usage

split = 'validation'
base_oci_path = 'oci://mosaicml-internal-datasets/scrolls-icl'
# perform tokenization to print out how many (max) tokens are present
# in the context
tokenize_lengths = True
# optionally upload to remote storage
upload = True

if tokenize_lengths:
    tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')

data_length = len(data[split])
for num, out_file in zip([10, 100, 500, len(data[split])], [
        f'/root/{dataset_name}_small.jsonl',
        f'/root/{dataset_name}_medium.jsonl',
        f'/root/{dataset_name}_500_samples.jsonl',
        f'/root/{dataset_name}_full.jsonl'
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

            context = data[split]['input'][i].strip()

            out = data[split]['output'][i].strip()

            row = prep_narrative(context, out)

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
        print(f'{max_token_length=}')
