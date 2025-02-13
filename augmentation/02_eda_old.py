from tqdm import tqdm
import argparse
import pandas as pd

from basic import ABSAAugmenter
from eda import *


class EDAAugmenter(ABSAAugmenter):
    def __init__(self, args):
        super(EDAAugmenter, self).__init__(args=args)

    def augment(self, eda_file):
        data = []
        with tqdm(total=self.dataloader.__len__()) as pbar:
            for i, inputs in enumerate(self.dataloader):
                source_text = inputs.source_text[0]
                if task == "tasd":
                    quads = [[q[0] for q in quad] for quad in inputs.quads]
                else:
                    quads = [[q[0] for q in quad] for quad in inputs.quads]       

                for j in range(self.args.num_aug):
                    index = j
                    valid = False
                    attempt = 0
                    max_attempts = 5000  # Maximale Versuche pro Beispiel
                    while not valid and attempt < max_attempts:
                  
                            # Augmentierte Sätze generieren
                            aug_sentence = eda(source_text,
                                                alpha_sr=self.args.alpha_sr,
                                                alpha_ri=self.args.alpha_ri,
                                                alpha_rs=self.args.alpha_rs,
                                                p_rd=self.args.alpha_rd,
                                                num_aug=self.args.num_aug)[index]
                            
                            if task == "tasd":
                                quads_upt = [[q[1], q[0], q[3]] for q in quads]
                            if task == "asqp":
                                quads_upt = [[q[1], q[0], q[3], q[2]] for q in quads]
                           
                            
                            # Formatieren der Daten im gewünschten Format
                            data.append(f"{aug_sentence}####{quads_upt}")
                            valid = True  # Erfolgreiche Augmentierung

                
                pbar.update(1)

        # Speichern in eine Textdatei
        with open(eda_file, 'w', encoding='utf_8_sig') as f:
            for line in data:
                f.write(line + '\n')

        print('Saved to -> [%s]' % eda_file)

import argparse
import itertools
import os
from argparse import Namespace


n_few_shot = [10, 50]
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["asqp", "tasd"]

combinations = itertools.product(n_few_shot, datasets, tasks)

for fs, dataset_name, task in combinations:
        file_path = f"augmentation/generations/eda/{task}_{dataset_name}_{fs}.txt"
        
        if not os.path.exists(file_path):
            args = {
                "n_few_shot": fs,
                "data_dir": dataset_name,
                "task": task,
                "num_aug": int(2000/fs),
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
