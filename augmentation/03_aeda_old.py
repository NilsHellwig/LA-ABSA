from tqdm import tqdm
import argparse
import pandas as pd
import itertools
import os
from argparse import Namespace

from basic import ABSAAugmenter
from aeda import *


class AEDAAugmenter(ABSAAugmenter):
    def __init__(self, args):
        super(AEDAAugmenter, self).__init__(args=args)

    def augment(self, aeda_file):
        data = []
        with tqdm(total=self.dataloader.__len__()) as pbar:
            for i, inputs in enumerate(self.dataloader):
                source_text = inputs.source_text[0]
                quads = [[q[0] for q in quad] for quad in inputs.quads]

                # augmented
                for _ in range(self.args.num_aug):
                    aug_sentence = aeda(source_text, punc_ratio=self.args.punc_ratio)

                    # Formatieren der Daten im gewünschten Format
                    data.append(f"{aug_sentence}####{quads}")

                pbar.update(1)

        # Speichern in eine Textdatei
        with open(aeda_file, 'w', encoding='utf_8_sig') as f:
            for line in data:
                f.write(line + '\n')

        print('Saved to -> [%s]' % aeda_file)


if __name__ == '__main__':
    n_few_shot = [10, 50]
    datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
    tasks = ["asqp", "tasd"]

    combinations = itertools.product(n_few_shot, datasets, tasks)

    for fs, dataset_name, task in combinations:
        file_path = f"augmentation/generations/aeda/{task}_{dataset_name}_{fs}.txt"

        if not os.path.exists(file_path):
            args = {
                "n_few_shot": fs,
                "data_dir": dataset_name,
                "task": task,
                "num_aug": int(2000/fs),
                "punc_ratio": 0.3
            }
            args = Namespace(**args)

            augmenter = AEDAAugmenter(args)
            augmenter.augment(aeda_file=file_path)
        else:
            print(f"Skipping: {file_path} already exists.")
