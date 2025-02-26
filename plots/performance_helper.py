from collections import Counter

from dataloader import DataLoader
dataloader = DataLoader("../datasets")

from evaluation import compute_f1_scores_quad, compute_scores_acd, compute_scores_acsa, compute_scores_single, count_regenerations
import json

from collections import defaultdict
import numpy as np

def limit_decimal_points(num_str, n_decimal=2):
    # Convert the string to a float and format it with the specified number of decimal places
    return f"{float(num_str):.{n_decimal}f}"

def calculate_mean_scores(scores_split_merge):
    # Initialisiere ein Dictionary für die Summen
    sums = {key: 0 for key in scores_split_merge[0].keys()}
    
    # Addiere die Werte für jeden Schlüssel in allen Dictionaries
    for score in scores_split_merge:
        for key in score:
            sums[key] += score[key]
    
    # Berechne den Mittelwert für jeden Schlüssel
    scores_split_merge_dict = {key: sums[key] / len(scores_split_merge) for key in sums}
    
    return scores_split_merge_dict


def to_array(pred_txt):
    if type(pred_txt) == list:
        return pred_txt
    else:
        return eval(pred_txt)
                    
def convert_json_to_tuple(aspects):
    if len(aspects[0]) == 3:
        return [(aspect["aspect_term"], aspect["aspect_category"], aspect["polarity"]) for aspect in aspects]
    if len(aspects[0]) == 4:
        return [(aspect["aspect_term"], aspect["aspect_category"], aspect["polarity"], aspect["opinion_term"]) for aspect in aspects]
                
def convert_array_to_tuple(aspects):
    if len(aspects) == 0:
        return []
    if len(aspects[0]) == 3:
        return [(aspect[0], aspect[1], aspect[2]) for aspect in aspects]
    if len(aspects[0]) == 4:
        return [(aspect[0], aspect[1], aspect[2], aspect[3]) for aspect in aspects]
    
def count_aspects(lst):
    # Initialisiere ein leeres Dictionary für die Zählungen
    element_count = {}
    
    # Iteriere durch die Liste
    for item in lst:
        # Erhöhe den Zähler für jedes Element
        if item in element_count:
            element_count[item] += 1
        else:
            element_count[item] = 1
    
    return element_count

def get_frequency_for_counts(counts, minimum):
    return sorted(counts, reverse=True)[0:minimum][minimum-1]

def get_unique_keys(dict_list):
    unique_keys = set()  # Set für einzigartige Schlüssel

    for d in dict_list:
        unique_keys.update(d.keys())  # Füge die Schlüssel zum Set hinzu

    return list(unique_keys)  # Wandle das Set in eine Liste um und gebe es zurück

def merge_aspect_lists(aspect_lists, minimum_appearance=3):
    
    aspect_lists_counter = []
    for aspect_list in aspect_lists:
        aspect_counter = dict(Counter([",".join(aspect) for aspect in aspect_list]))
        aspect_lists_counter.append(aspect_counter)
        
    unique_tuples = get_unique_keys(aspect_lists_counter)

    label = []
    for tuple_str in unique_tuples:

        count_tuple =  get_frequency_for_counts([asp.get(tuple_str, 0) for asp in aspect_lists_counter], minimum_appearance)
        tuple_reverse = tuple(tuple_str.split(","))
        
        label += count_tuple * [tuple_reverse]

    return label


def get_performance_scores(task, dataset, model, mode, evaluation_target, n_shots=0, self_consistency=False, minimum_appearance=3, n_seeds=5, evaluation_type="quad", range_seed=None):
    if evaluation_target == "cross_validation":
        ds_list = ["train", "test", "dev"]
    if evaluation_target == "test":
        ds_list = ["test"]
        
    if range_seed == None:
        range_seed = range(0, n_seeds)
    else:
        range_seed = range(range_seed[0], range_seed[1])
        
    # 1. Load Dataset for seed
    ds_parts = []
    for seed in range_seed:
        ds_parts.append([])
        for ds_type in ds_list:
            with open(
                f"../generations/zeroshot/{task}_{dataset}_{ds_type}_{model}_{seed}_{mode}_{n_shots}.json",
                "r",
                encoding="utf-8",
            ) as f:
                data = json.load(f)
            ds_parts[seed] += data

            
        
    # 2. Split Dataset in 5 parts <---- hier schauen, ob es die gleichen 5 parts sind
    if evaluation_target == "cross_validation":
        splits = [dataloader.random_cross_validation_split(ds) for ds in ds_parts]
    if evaluation_target == "test":
        splits = [[d] for d in ds_parts]


    # 3. Calculate Performance for each split: without self-consistency
    if self_consistency == False:
        scores_split_merge = []
        for split_idx in range(len(splits[0])):

            # combine datasets
            seed_split_merge = []
            for seed in range_seed:

                pred_pt = [
                    convert_array_to_tuple(to_array(sp["pred_label"]))
                    for sp in splits[seed][split_idx]
                ]
                gold_pt = [convert_json_to_tuple(sp["aspects"]) for sp in splits[seed][split_idx]]
                
                
                load_dataset_unique_ac = dataset if dataset != "gerest" else "rest16" #### EVTL noch löschen!!!!!!
                unique_aspect_categories = list({aspect['aspect_category'] for entry in dataloader.load_data(name=load_dataset_unique_ac, data_type="all", target=task) for aspect in entry['aspects']})
                
                if evaluation_type == "quad":
                    scores = compute_f1_scores_quad(pred_pt, gold_pt)
                elif evaluation_type == "regeneration_count":
                    scores = count_regenerations(splits[seed][split_idx])
                elif evaluation_type == "acd":
                    scores = compute_scores_acd(pred_pt, gold_pt, unique_aspect_categories)
                elif evaluation_type == "acsa":
                    scores = compute_scores_acsa(pred_pt, gold_pt, unique_aspect_categories)
                elif "single" in evaluation_type:
                    scores = compute_scores_single(pred_pt, gold_pt, evaluation_type)
                seed_split_merge.append(scores)

            seed_split_merge_dict = calculate_mean_scores(seed_split_merge)
            scores_split_merge.append(seed_split_merge_dict)

    else:  # 3. With self-consistency
        scores_split_merge = []
        for split_idx in range(len(splits[0])):

            # combine datasets
            pred_pt_total = []
            
            # gold_pt ist gleich, egal, welchen seed ich betrachte, hier beispielhaft den 1.
            gold_pt = [convert_json_to_tuple(sp["aspects"]) for sp in splits[0][split_idx]]
            
                
            for seed in range_seed:
                pred_pt = [
                    convert_array_to_tuple(to_array(sp["pred_label"]))
                    for sp in splits[seed][split_idx]
                ]

                pred_pt_total.append(pred_pt)
                           
            consistency_merged_pred_pt = [merge_aspect_lists([pred_pt_total[seed][example_idx] for seed in range_seed], minimum_appearance=minimum_appearance) for example_idx in range(len(pred_pt_total[0]))]

            if evaluation_type == "quad":
                scores = compute_f1_scores_quad(consistency_merged_pred_pt, gold_pt)
            elif evaluation_type == "regeneration_count":
                scores = count_regenerations(splits[seed][split_idx])
            elif evaluation_type == "acd":
                scores = compute_scores_acd(consistency_merged_pred_pt, gold_pt, unique_aspect_categories)
            elif evaluation_type == "acsa":
                scores = compute_scores_acsa(consistency_merged_pred_pt, gold_pt, unique_aspect_categories)
            elif "single" in evaluation_type:
                scores = compute_scores_single(consistency_merged_pred_pt, gold_pt, evaluation_type)
            scores_split_merge.append(scores)
            
    
    scores_split_merge_dict = calculate_mean_scores(scores_split_merge)

    return scores_split_merge_dict

# Methods ASQP ACOS TASD ASTE
# R15 R16 Lap Rest R15 R16 L14 R14 R15 R16 AVG
# TAS-BERT (Wan et al., 2020) 34.78 43.71 27.31 33.53 57.51 65.89 - - - - -
# Jet-BERT (Xu et al., 2020) - - - - - - 51.04 62.40 57.53 63.83 -
# Extract-Classify (Cai et al., 2021) 36.42 43.77 35.80 44.61 - - - - - - -
# GAS (Zhang et al., 2021c) 45.98 56.04 - - 60.63 68.31 58.19 70.52 60.23 69.05 -
# Paraphrase (Zhang et al., 2021b) 46.93 57.93 43.51 61.16 63.06 71.97 61.13 72.03 62.56 71.70 61.20
# UIE (Lu et al., 2022b) - - - - - - 62.94 72.55 64.41 72.86 -
# Seq2Path (Mao et al., 2022) - - 42.97 58.41 63.89 69.23 64.82 75.52 65.88 72.87 -
# DLO (Hu et al., 2022) 48.18 59.79 43.64 59.99 62.95 71.79 61.46 72.39 64.26 73.03 61.75
# UnifiedABSA†
# (Wang et al., 2022c) - - 42.58 60.60 - - - - - - -
# LEGO-ABSA†
# (Gao et al., 2022) 46.10 57.60 - - 62.30 71.80 62.20 73.70 64.40 69.90 -
# MVP 51.04 60.39 43.92 61.54 64.53 72.76 63.33 74.05 65.89 73.48 63.09
# MVP (multi-task)† 52.21 58.94 43.84 60.36 64.74 70.18 65.30 76.30 69.44 73.10 63.44

# Methods Rest15 Rest16
# Pre Rec F1 Pre Rec F1
# HGCN-BERT+BERT-Linear∗
# (Cai et al., 2020) 24.43 20.25 22.15 25.36 24.03 24.68
# HGCN-BERT+BERT-TFM∗
# (Cai et al., 2020) 25.55 22.01 23.65 27.40 26.41 26.90
# TASO-BERT-Linear∗
# (Wan et al., 2020) 41.86 26.50 32.46 49.73 40.70 44.77
# TASO-BERT-CRF∗
# (Wan et al., 2020) 44.24 28.66 34.78 48.65 39.68 43.71
# Extract-Classify-ACOS (Cai et al., 2021) 35.64 37.25 36.42 38.40 50.93 43.77
# GAS∗
# (Zhang et al., 2021b) 45.31 46.70 45.98 54.54 57.62 56.04
# Paraphrase∗
# (Zhang et al., 2021a) 46.16 47.72 46.93 56.63 59.30 57.93
# DLO 47.08 49.33 48.18 57.92 61.80 59.79
# ILO 47.78 50.38 49.05 57.58 61.17 59.32


def get_finetuned_scores(task="asqp", dataset="rest16", method="mvp", n_seeds=5, n_shots=None):
    if dataset in ["rest15", "rest16"] and n_shots is None:
        with open("../plots/past_results.json", "r", encoding="utf-8") as f:
            past_results = json.load(f)
        
        # Zugriff auf die Ergebnisse prüfen
        results = past_results[task][method][dataset]
        return results
    
    merged_scores = defaultdict(list)
    
    for seed in range(0, n_seeds):
        if n_shots is None:
            with open(f"../generations/00_baselines/training_{task}_{dataset}_seed-{seed}_n-train_{method}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(f"../generations/00_baselines/training_{task}_{dataset}_seed-{seed}_n-train_{method}_{n_shots}.json", "r", encoding="utf-8") as f:
                data = json.load(f)
    
        scores = compute_f1_scores_quad(data["all_preds"], data["all_labels"])
    
        for key, value in scores.items():
            merged_scores[key].append(value)

    merged_scores = {key: np.mean(values) for key, values in dict(merged_scores).items()}

    return merged_scores