#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluation."""

from ..arg_metav_formatter import arg_metav_formatter
from seqeval.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import os
import json
import argparse


def read_tag(file):
    labels = []
    example = []
    for line in open(file, 'r'):
        line = line.strip()
        if line != '':
            example.append(line)
        else:
            labels.append(example)
            example = []
    return labels


def read_label(file):
    return [line.strip() for line in open(file)]


def f1(labels, predictions, language=None):
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    return {
        'f1': f1 * 100,
        'precision': precision * 100,
        'recall': recall * 100
    }


def accuracy(labels, predictions, language=None):
    correct = sum([int(p == l) for p, l in zip(predictions, labels)])
    accuracy = float(correct) / len(predictions)
    return {'accuracy': accuracy * 100}


GROUP2TASK = {
    "classification": ["pawsx", "xnli"],
}

TASK2LANGS = {
    "pawsx": "de,en,es,fr,ja,ko,zh".split(","),
}

READER_FUNCTION = {
    'pawsx': read_label,
}

METRIC_FUNCTION = {
    'pawsx': f1,
}


def evaluate_one_task(prediction_file, label_file, task, language=None):
    """Evalute the classification tasks by accuracy.
  Args:
    prediction_file (string): path to the prediction tsv file.
    label_file (string): path to the grouth truth tsv file.
  Return:
    result (dict): a dictionary with accuracy.

  Both input files contain one example per line as follows:
    ``[label]\t[sentence1]\t[sentence2]``
  """
    predictions = READER_FUNCTION[task](prediction_file)
    labels = READER_FUNCTION[task](label_file)
    if task not in ['bucc2018', 'mlqa', 'tydiqa', 'xquad']:
        assert len(predictions) == len(
            labels
        ), 'Number of examples in {} and {} not matched in {} task'.format(
            prediction_file, label_file, task)
    result = METRIC_FUNCTION[task](labels, predictions, language)
    return result


def evaluate(prediction_folder, label_folder, verbose=False):
    """Evaluate on all tasks if available.
  Args:
    prediction_folder (string): prediction folder that contains each task's prediction in each subfolder.
    label_file (string): label folder that contains each task's ground-truth label in each subfolder.
  Return:
    overall_scores (dict): a dictionary with sub-group scores. key: group label.
    detailed_scores (dict): a dictionary with all detailed scores. key: task label.
  """
    prediction_tasks = next(os.walk(prediction_folder))[1]
    label_tasks = next(os.walk(label_folder))[1]
    # prediction_tasks = label_tasks = ['mlqa', 'tydiqa', 'xquad']

    detailed_scores = {}
    for task, langs in TASK2LANGS.items():
        if task in prediction_tasks and task in label_tasks:
            suffix = "json" if task in GROUP2TASK["qa"] else "tsv"
            # collect scores over all languages
            score = defaultdict(dict)
            for lg in langs:
                prediction_file = os.path.join(prediction_folder, task,
                                               f"test-{lg}.{suffix}")
                label_file = os.path.join(label_folder, task,
                                          f"test-{lg}.{suffix}")
                score_lg = evaluate_one_task(prediction_file,
                                             label_file,
                                             task,
                                             language=lg)
                for metric in score_lg:
                    score[metric][lg] = score_lg[metric]
            # average over all languages
            avg_score = {}
            for m in score:
                avg_score[f'avg_{m}'] = sum(score[m].values()) / len(score[m])
            score.update(avg_score)
            if task in GROUP2TASK["qa"]:
                score['avg_metric'] = (score['avg_exact_match'] +
                                       score['avg_f1']) / 2
            elif 'avg_f1' in score:
                score['avg_metric'] = score['avg_f1']
            elif 'avg_accuracy' in score:
                score['avg_metric'] = score['avg_accuracy']
            detailed_scores[task] = score
            if verbose:
                avg_result = ', '.join([
                    '{}={:.1f}'.format(k, v) for k, v in score.items()
                    if k.startswith('avg')
                ])
                print('- Evaluate {}:\t{}'.format(task, avg_result))

    # Display logic:
    overall_scores = {}
    all_tasks = set(TASK2LANGS.keys())
    available_tasks = set(detailed_scores.keys())

    # If scores of all tasks are available, show the overall score in the main table
    if all_tasks == available_tasks:
        overall_scores['all_task'] = sum(
            detailed_scores[task]['avg_metric']
            for task in all_tasks) / len(all_tasks)

    # If scores of all tasks in a sub group are available, show the score in the sub table
    for group, group_tasks in GROUP2TASK.items():
        if len(set(group_tasks) - available_tasks) == 0:
            overall_scores[group] = sum(
                detailed_scores[task]['avg_metric']
                for task in group_tasks) / len(group_tasks)

    return overall_scores, detailed_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument('--prediction_folder',
                        default=None,
                        type=str,
                        required=True,
                        help='the predictions of one model')
    parser.add_argument('--label_folder',
                        default=None,
                        type=str,
                        required=True,
                        help='the grouth truth file')
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='whether to print details')
    args = parser.parse_args()
    overall_scores, detailed_scores = evaluate(args.prediction_folder,
                                               args.label_folder, args.verbose)
    overall_scores.update(detailed_scores)
    print(json.dumps(overall_scores))
