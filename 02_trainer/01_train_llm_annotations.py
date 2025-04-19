import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../zero-shot-absa-quad/')))
from dataloader import DataLoader
from trainer import train_paraphrase, train_mvp, train_dlo
import json
import time

from helper import clean_up, create_output_directory

dataloader = DataLoader("../zero-shot-absa-quad/datasets", "../zero-shot-absa-quad/fs_examples")


for seed in range(5):
 for ds_name in ["rest16", "hotels", "rest15", "flightabsa", "coursera"]:
  for fs_num in [50, 10, 0]:
   for task in ["tasd", "asqp"]:
     for n_llm_examples in [800, "full"]:  
         train_ds = dataloader.load_data(ds_name, "train", cv=False, target=task, fs_num=fs_num, fs_ann_mode=True, n_ann_examples=n_llm_examples)
         test_ds = dataloader.load_data(ds_name, "test", cv=False, target=task)
      
         for ml_method in ["paraphrase", "dlo"]:
            print(f"Task:", task, "Dataset:", ds_name, "Seed:", seed, "ML-Method:", ml_method, "FS-Num:", fs_num, "len(train_ds)", len(train_ds), "len(test_ds)", len(test_ds), "n_llm_examples", n_llm_examples)
            filename = f"./_out_fine_tunings/01_llm_annotate_train/{ml_method}_{n_llm_examples}_{task}_{fs_num}_{ds_name}_{seed}.json"

            if os.path.exists(filename):
               print(f"File {filename} already exists. Skipping.")
               continue
            else:
            
               clean_up()
               create_output_directory()
              
               if ml_method == "paraphrase":
                  scores = train_paraphrase(train_ds=train_ds, test_ds=test_ds, seed=seed, dataset=ds_name, task=task)
               if ml_method == "mvp":
                  scores = train_mvp(train_ds=train_ds, test_ds=test_ds, seed=seed, dataset=ds_name, task=task)
               if ml_method == "dlo":
                  scores = train_dlo(train_ds=train_ds, test_ds=test_ds, seed=seed, dataset=ds_name, task=task)
              
    
               with open(filename, 'w', encoding='utf-8') as json_file:
                  json.dump(scores, json_file, ensure_ascii=False, indent=4)
                  
               