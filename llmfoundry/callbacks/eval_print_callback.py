# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Print Evals"""
from __future__ import annotations

import torch
from composer.core import Callback, State
from composer.loggers import Logger

class PrintBatchContents(Callback):

    def __init__(self, num_consecutive_batches_printed: int = 1):
        self.batch_print_counter = 0
        self.num_consecutive_batches_printed = num_consecutive_batches_printed


    def eval_start(self, state: State, logger: Logger):
        self.batch_print_counter = 0

    def eval_batch_start(self, state: State, logger: Logger):

        if self.batch_print_counter < self.num_consecutive_batches_printed:
            print('\033[92m' + 'Example: ' + '\033[0m')
            print("-" * 10)
            example = state.batch['input_ids'][0]
            print(state.model.tokenizer.decode(example, skip_special_tokens=True))
            # print(state.batch)
            # if 'continuation_indices' in state.batch:
                # continuation_indices = list(state.batch['continuation_indices'][0])
                # print('\033[94m' + 'Continuation Indices: ' + '\033[0m')
                # print(state.model.tokenizer.decode(example[continuation_indices[0]:continuation_indices[-1]], skip_special_tokens=True))
            self.batch_print_counter += 1
            print('-' * 10)
