import os
import ast
import random
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import json
from helper import merge_aspect_lists

class DataLoader:
    def __init__(self, base_path="datasets", fs_path="fs_examples"):
        self.base_path = base_path
        self.fs_path = fs_path
        
    def load_fs_ann(self, name, data_type, target, fs_num, llm_name):
        
        data = []
        for seed in range(5):
          with open(f"./generations/llm_annotations/{target}_{name}_train_{llm_name}_{seed}_label_{fs_num}.json", "r", encoding="utf-8") as file:
            examples = json.load(file)
          data.append(examples)
        
        lines = []
        for k in range(len(data[0])):
            labels = []

            for i in range(0, len(data)):
                if data[i][k]["pred_label"] != []:
                   labels += [data[i][k]["pred_label"]]
                else:
                   labels.append([])
            merged_label = merge_aspect_lists(labels, minimum_appearance=3)

            
            for l in merged_label:
                if len(l) > 4:
                    raise KeyboardInterrupt(merged_label)
            if len(merged_label) > 0:
               lines.append(f"{data[0][k]['text']}####{merged_label}")
            
        return lines

        
    def load_data(self, name, data_type, cv=False, seed=42, target="asqp", fs_mode=False, fs_num=0, fs_ann_mode=False, llm_name="gemma2:27b", n_ann_examples="full"):
        if fs_mode or fs_ann_mode:
            dataset_paths = [os.path.join(self.fs_path, target, name, f"fs_{str(fs_num)}", "examples.txt")] 
        else:
            dataset_paths = ["train", "test", "dev"] if data_type == "all" else [data_type]
            dataset_paths = [os.path.join(self.base_path, target, name, f"{d_path}.txt") for d_path in dataset_paths]

        if fs_ann_mode:
            dataset_paths += [f"../generations/llm_annotations/"]
            
        data = []

        for d_path in dataset_paths:
            
            lines = []
            
            if "generations/llm_annotations" in d_path:
                    lines += self.load_fs_ann(name, data_type, target, fs_num, llm_name)
                    if n_ann_examples != "full":
                        lines = lines[0:n_ann_examples]
                    lines = lines[0:len(lines)-fs_num]
                        
            else:
                    with open(d_path, 'r', encoding='utf-8') as file:
                       lines += file.readlines()
                    
                    
            for idx, line in enumerate(lines):
                try:
                    text, aspects_str = line.split("####")
                    aspects = ast.literal_eval(aspects_str.strip())
                    aspect_list = []
                

                    for aspect in aspects:
                        aspect_dict = {
                            "aspect_term": aspect[0],
                            "aspect_category": aspect[1],
                            "polarity": aspect[2]
                        }
                        # Add 'opinion_term' only if target is 'asqp'
                        if target == "asqp":
                            aspect_dict["opinion_term"] = aspect[3]
                        aspect_list.append(aspect_dict)

                    if len(aspects) > 0:
                        data.append({
                            "id": f"{idx}_{name}_{d_path}",
                            "text": text.strip(),
                            "aspects": aspect_list,
                            "tuple_list": [tuple(aspect) for aspect in aspects]
                        })
                except ValueError as e:
                    print(f"Skipping line {idx} in {d_path} due to ValueError: {e}")
                    continue
        
        if cv:
            return self.random_cross_validation_split(data, seed)
        
        return data




    def random_cross_validation_split(self, data, seed=42):
        categories = [[el["aspect_category"] for el in an["aspects"]] for an in data]

        mlb = MultiLabelBinarizer()
        Y = mlb.fit_transform(categories)

        n_splits = 5
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
        splits = []
        for train_index, test_index in mskf.split(np.zeros(len(Y)), Y):
           splits.append([data[i] for i in test_index])
    
    
        return splits