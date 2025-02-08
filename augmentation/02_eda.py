from tqdm import tqdm
import argparse
import pandas as pd

from basic import ABSAAugmenter
from eda import *


class EDAAugmenter(ABSAAugmenter):
    def __init__(self, args):
        super(EDAAugmenter, self).__init__(args=args)

    def augment(self, eda_file):
        data = {'quads': [], 'text': []}
        with tqdm(total=self.dataloader.__len__()) as pbar:
            for i, inputs in enumerate(self.dataloader):
                source_text = inputs.source_text[0]
                quads = [[q[0] for q in quad] for quad in inputs.quads]

                # # original
                # data['quads'].append(quads)
                # data['text'].append(source_text)

                # augmented
                try:
                    aug_sentences = eda(source_text,
                                        alpha_sr=self.args.alpha_sr,
                                        alpha_ri=self.args.alpha_ri,
                                        alpha_rs=self.args.alpha_rs,
                                        p_rd=self.args.alpha_rd,
                                        num_aug=self.args.num_aug)
                except:
                    # too short to EDA
                    continue

                for aug_sentence in aug_sentences:
                    data['quads'].append(quads)
                    data['text'].append(aug_sentence)

                pbar.update(1)
        df = pd.DataFrame.from_dict(data)
        df.to_csv(eda_file, encoding='utf_8_sig', index=False)
        print('Saved to -> [%s]' % eda_file)

import argparse
import itertools
import os
from argparse import Namespace


n_few_shot = [10, 50]
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["asqp"]

combinations = itertools.product(n_few_shot, datasets, tasks)

for fs, dataset_name, task in combinations:
        file_path = f"augmentation/generations/eda/{task}_{dataset_name}_{fs}.txt"
        
        if not os.path.exists(file_path):
            args = {
                "n_few_shot": fs,
                "data_dir": dataset_name,
                "task": task,
                "num_aug": 5,
                "alpha_sr": 0.1,
                "alpha_ri": 0.1,
                "alpha_rs": 0.1,
                "alpha_rd": 0.1
            }
            args = Namespace(**args)
            
            augmenter = EDAAugmenter(args)
            augmenter.augment(eda_file=file_path)
        else:
            print(f"Skipping: {file_path} already exists.")
