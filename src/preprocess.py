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

from __future__ import absolute_import, division, print_function
from .arg_metav_formatter import arg_metav_formatter
import argparse
import os
import csv


def pawsx_preprocess(args):
    def _preprocess_one_file(infile):
        data = []
        for i, line in enumerate(open(infile, 'r')):
            if i == 0:
                continue
            items = line.strip().split('\t')
            sent1 = ' '.join(items[1].strip().split(' '))
            sent2 = ' '.join(items[2].strip().split(' '))
            label = items[3]
            data.append([sent1, sent2, label])
        with open(outfile, 'w') as fout:
            writer = csv.writer(fout, delimiter='\t')
            for sent1, sent2, label in data:
                writer.writerow([sent1, sent2, label])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    split2file = {'train': 'train', 'test': 'test_2k', 'dev': 'dev_2k'}
    for lang in ['en', 'de', 'es', 'fr', 'ja', 'ko', 'zh']:
        for split in ['train', 'test', 'dev']:
            file = split2file[split]
            infile = os.path.join(args.data_dir, lang, "{}.tsv".format(file))
            outfile = os.path.join(args.output_dir,
                                   "{}-{}.tsv".format(split, lang))
            _preprocess_one_file(infile, outfile)
            print(f'finish preprocessing {outfile}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=arg_metav_formatter)
    parser.add_argument("--data_dir",
                        default="./data/x-final",
                        type=str,
                        help="The input data directory containing tsv files")
    parser.add_argument("--output_dir",
                        default="./data/paws_x",
                        type=str,
                        help="The output data directory")
    args = parser.parse_args()
    pawsx_preprocess(args)
