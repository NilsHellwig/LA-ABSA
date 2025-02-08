import torch.cuda
from tqdm import tqdm
import argparse
import pandas as pd
import nlpaug.augmenter.word as naw
import random
import itertools
import os
from argparse import Namespace


from basic import ABSAAugmenter


class BTAugmenter(ABSAAugmenter):
    def __init__(self, args, candidate_langs=['fr', 'zh', 'ar', 'jap', 'da']): # jap
        super(BTAugmenter, self).__init__(args=args)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.candidate_langs = candidate_langs

        # assert (self.args.num_aug == len(candidate_langs))

        self.all_back_translation_aug = []
        for lang in candidate_langs:
            self.all_back_translation_aug.append(naw.BackTranslationAug(
                    from_model_name='Helsinki-NLP/opus-mt-en-%s' % lang,
                    to_model_name='Helsinki-NLP/opus-mt-%s-en' % lang,
                    device=device
                )
            )

    def augment(self, bt_file):
        data = []
        with tqdm(total=self.dataloader.__len__()) as pbar:
            for i, inputs in enumerate(self.dataloader):
                source_text = inputs.source_text[0]
                quads = [[q[0] for q in quad] for quad in inputs.quads]
                
                for j in range(self.args.num_aug):
                    index = j
                    valid = False
                    attempt = 0
                    max_attempts = 5000  # Maximale Versuche pro Beispiel
                    while not valid and attempt < max_attempts:
                        try:
                            aug_sentence = self.all_back_translation_aug[index].augment(source_text)[0]
                            data.append(f"{aug_sentence}####{quads}")
                            valid = True  # Erfolgreiche Augmentierung
                        except:
                            attempt += 1  # Erhöhe den Versuchszähler
                            continue  # Falls Fehler, erneut versuchen
                
                pbar.update(1)
        
        with open(bt_file, 'w', encoding='utf_8_sig') as f:
            for line in data:
                f.write(line + '\n')
        
        print('Saved to -> [%s]' % bt_file)



n_few_shot = [10, 50] # 0 fehlt noch
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["asqp"]

n_generations = 100


combinations = itertools.product(n_few_shot, datasets, tasks)

for combination in combinations:
    fs,  dataset_name, task = combination
    file_path = f"augmentation/generations/back_translation/{task}_{dataset_name}_{fs}.txt"
    # Prüfen, ob die Datei bereits existiert
    if not os.path.exists(file_path):
        args = {"data_dir": dataset_name, "task": task, "num_aug": 5, "n_few_shot": fs}
        args = Namespace(**args)
        augmenter = BTAugmenter(args)
        augmenter.augment(bt_file=file_path)

    else:
        print(f"Skipping: {file_path} already exists.")