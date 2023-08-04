# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Periodically log generations to wandb from a set of prompts."""
import copy
import re
from collections import defaultdict
from typing import Union

from composer.core import Callback, Event, State
from composer.loggers import Logger
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


class AverageICLLogger(Callback):

    def run_event(self, event: Event, state: State, logger: Logger):
        if event != Event.FIT_START and event != Event.EVAL_AFTER_ALL:
            return

        eval_metrics = copy.deepcopy(state.eval_metrics)
        num_shot_avgs = defaultdict(list)
        for metric_name, metrics in eval_metrics.items():
            for _, metric_val in metrics.items():
                match = re.search(r'(\d+)-shot', metric_name)
                if not match:
                    continue
                num_shots = int(match.group(1))
                num_shot_avgs[num_shots].append(metric_val.compute())
        num_shot_avgs = {
            f'metrics/icl/{num_shot}-shot/avg': sum(perfs) / len(perfs)
            for num_shot, perfs in num_shot_avgs.items()
        }
        logger.log_metrics(num_shot_avgs)
