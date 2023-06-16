# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Print Evals."""
from __future__ import annotations

import torch
from composer.core import Callback, State
from composer.loggers import Logger


class PrintEvalExample(Callback):
    """Prints the first example in an eval batch.

    Prints the first example in an eval batch
    Args:
        num_consecutive_batches_to_print (int): allows user to print
        multiple examples from consecutive batches
    """

    def __init__(self, num_consecutive_batches_to_print: int = 1):
        self.batches_printed = 0
        self.num_consecutive_batches_to_print = num_consecutive_batches_to_print

    def eval_start(self, state: State, logger: Logger):
        # reset batch counter between different eval tasks
        self.batches_printed = 0

    def eval_batch_start(self, state: State, logger: Logger):

        if self.batches_printed < self.num_consecutive_batches_to_print:
            print('\033[92m' + 'Example: ' + '\033[0m')
            print('-' * 10)
            example = state.batch['input_ids'][0]
            if state.is_model_ddp:
                tokenizer = state.model.module.tokenizer
            else:
                tokenizer = state.model.tokenizer
            print(
                tokenizer.decode(example, skip_special_tokens=True))
            self.batches_printed += 1
            print('-' * 10)
