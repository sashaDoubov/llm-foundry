# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import re
import string
from collections import Counter
from typing import List

import torch
from composer.metrics.nlp import InContextLearningQAAccuracy
from torch import Tensor


class InContextLearningQAF1(InContextLearningQAAccuracy):
    r"""Computes accuracy for In-context learning (ICL) question answering (QA)
    tasks.

    ICL QA tasks consist of some number of example question answering tasks (referred to as the 'context'), followed by a test task where the model must
    match one of the possible answer aliases (referred to as the 'continuation').

    For example, the model may be provided the context below and evaluated on its ability to correctly predict the continuation.

    Context: `Question: Who was president of the United States in 2012?\nAnswer: Barack Obama\nQuestion: Is water wet?\nAnswer: `
    Continuation: [`yes`, `no`]

    Both predictions and answers will be normalized before comparison.

    Adds metric state variables:
        correct (float): The number of instances where the prediction was a prefix for any of the answer aliases.
        total (float): The number of total instances that were predicted.

    Args:
        dist_sync_on_step (bool, optional): Synchronize metric state across processes at
            each forward() before returning the value at the step. Default: ``False``.
    """

    # Make torchmetrics call update only once
    full_state_update = False

    def __init__(self, dist_sync_on_step: bool = False):
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state('tp', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('precision_denom',
                       default=torch.tensor(0.),
                       dist_reduce_fx='sum')
        self.add_state('recall_denom',
                       default=torch.tensor(0.),
                       dist_reduce_fx='sum')

    def normalize_answer(self, answer: str):
        """Lower text and remove punctuation, articles and extra whitespace.

        Copied from https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py
        """

        def remove_articles(text: str) -> str:
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text: str) -> str:
            return ' '.join(text.split())

        def handle_punc(text: str) -> str:
            exclude = set(string.punctuation +
                          ''.join([u'‘', u'’', u'´', u'`']))
            return ''.join(ch if ch not in exclude else ' ' for ch in text)

        def lower(text: str) -> str:
            return text.lower()

        def replace_underscore(text: str) -> str:
            return text.replace('_', ' ')

        return white_space_fix(
            remove_articles(handle_punc(lower(
                replace_underscore(answer))))).strip()

    def update(self, outputs: List[str], labels: List[List[str]]):
        for sample_output, sample_labels in zip(outputs, labels):
            cleaned_sample_output_tokens = self.normalize_answer(
                sample_output).split()
            cleaned_sample_labels_tokens = self.normalize_answer(
                sample_labels[0]).split()
            common = Counter(cleaned_sample_output_tokens) & Counter(
                cleaned_sample_labels_tokens)

            num_same = sum(common.values())

            if num_same == 0:
                continue

            # print(f"{common=}")
            # print(f"{cleaned_sample_labels_tokens=}")
            # print(f"{cleaned_sample_output_tokens=}")

            self.tp += torch.tensor(num_same)

            self.precision_denom += torch.tensor(
                len(cleaned_sample_output_tokens))
            self.recall_denom += torch.tensor(len(cleaned_sample_labels_tokens))

    def compute(self):
        assert isinstance(self.tp, Tensor)
        assert isinstance(self.precision_denom, Tensor)
        assert isinstance(self.recall_denom, Tensor)
        precision = self.tp / self.precision_denom
        recall = self.tp / self.recall_denom
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
