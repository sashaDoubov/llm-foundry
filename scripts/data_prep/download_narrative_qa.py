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

data = load_dataset('tau/scrolls', 'narrative_qa')

random.seed(0)


def prep_narrative(context, answer):
    return {
        'context': context,
        'answer': answer,
    }


# Example usage

split = 'validation'
base_oci_path = 'oci://mosaicml-internal-datasets/scrolls-icl'
tokenize_lengths = True
upload = False

if tokenize_lengths:
    tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b')

data_length = len(data[split])
for num, out_file in zip([10, 100, 500, len(data[split])], [
        '/root/narrative_qa_small.jsonl', '/root/narrative_qa_medium.jsonl',
        '/root/narrative_qa_500_samples.jsonl', '/root/narrative_qa_full.jsonl'
]):
    if tokenize_lengths:
        max_token_length = 0
    print(data_length)
    if num < data_length:
        sequence = random.sample(list(range(data_length)), num)
    else:
        sequence = range(data_length)
    print(sequence)

    with open(out_file, 'w', encoding='utf8') as f:

        for iter_num, i in enumerate(tqdm.tqdm(sequence)):

            if iter_num < 5:
                print(i)

            context = data[split]['input'][i]

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
