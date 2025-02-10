import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from validator import validate_label, validate_reasoning
from promptloader import PromptLoader
from dataloader import DataLoader
from llm import LLM
import itertools
import json
import random


## Load API Key
from dotenv import load_dotenv
load_dotenv()
GWDG_KEY = os.getenv("GWDG_KEY")  

## LLM

dataloader = DataLoader()
promptloader = PromptLoader()

SPLIT_SEED = 42


def validate_augmentation(output, unique_aspect_categories, example_original):
    if not("####" in output):
        return False
    label_original = eval(example_original.split("####")[1])
    text_original = example_original.split("####")[0]
    label_original_comp = "".join(sorted(["-{}{}".format(label[1], label[2]) for label in label_original]))
    
    output = output.split("####")
    output = [o.strip() for o in output if o != ""]
    sentence = output[0]
    labels = output[1] + "]"
    try :
        labels = eval(labels)
        labels = [tuple(l) for l in labels]
    except:
        return False
    
    
    validate_labels = validate_label(str(labels), sentence, unique_aspect_categories=unique_aspect_categories)
    if validate_labels[0] == False:
        return False
    labels = validate_labels[0]
    
    # Check if sentiment and categories are in sentence
    labels_txt_comp = "".join(sorted(["-{}{}".format(label[1], label[2]) for label in labels]))
    
    if label_original_comp != labels_txt_comp:
        return False
    
    if text_original.strip() == sentence.strip():
        return False
    
    return sentence, labels
    

def create_annotations(TASK, DATASET_NAME, LLM_BASE_MODEL, SEED, N_FEW_SHOT, N_SYNTHETIC):
    

    print(f"TASK:", TASK)
    print(f"DATASET_NAME: {DATASET_NAME}")
    print(f"LLM_BASE_MODEL: {LLM_BASE_MODEL}")
    print(f"SEED: {SEED}")
    print(f"N_FEW_SHOT: {N_FEW_SHOT}")
    print(f"N_SYNTHETIC: {N_SYNTHETIC}")

    ## Load Model

    llm = LLM(LLM_BASE_MODEL, parameters=[
        {"name": "stop", "value": [")]"]}, 
        {"name": "num_ctx", "value": "4096"}
        ]) #8192
    
    ## Unique Aspect Categories

    unique_aspect_categories = sorted({aspect['aspect_category'] for entry in dataloader.load_data(name=DATASET_NAME, data_type="all", target=TASK) for aspect in entry['aspects']})
    if DATASET_NAME == "gerest" and not("food general" in unique_aspect_categories):
        unique_aspect_categories += ["food general"]
    unique_aspect_categories = sorted(unique_aspect_categories)
    predictions = []
    
    ## Load Examples
    
    with open(f"./fs_examples/{TASK}/{DATASET_NAME}/fs_{N_FEW_SHOT}/examples.txt", "r", encoding="utf-8") as file:
        few_shot_originals = file.readlines()  # Liste von Zeilen
        
    ## Load Prompt
    with open(f"./prompt/{TASK}/prompt-augmentation.txt", "r", encoding="utf-8") as file:
        prompt_original = file.read()
    prompt_original = prompt_original.replace("[[aspect_category]]", ", ".join(unique_aspect_categories))



    ## label
    valid_examples = []

    for idx, example in enumerate(few_shot_originals * int(N_SYNTHETIC / len(few_shot_originals))):
        seed = SEED
        prompt = prompt_original.replace("[[examples]]", example)

        correct_output = False   
        n_retries = 0
        print("######", idx, example)
        while not correct_output:
            output, duration = llm.predict(prompt, seed)
            print("output", output)
            correct_output = True

            validation_augmentation = validate_augmentation(output, unique_aspect_categories, example)
            if not validation_augmentation:
                correct_output = False
                n_retries += 1
                seed += 5
            else:
                sentence, labels = validation_augmentation
                valid_examples.append(f"{sentence}####{[list(l) for l in labels]}".strip())
                correct_output = True
        
            if n_retries > 10:
                valid_examples.append(example.strip())
                break

    with open(f"generations/augmentations/{TASK}_{DATASET_NAME}_{N_FEW_SHOT}.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(valid_examples) + "\n")

        
        

# todo: korrekt label formatieren in txt
# konstante für anzahl an synthetischen beispielen    
        
        

##### create annotations

# tasks = ["asqp", "tasd"]
# datasets = ["rest15", "rest16"]
# dataset_types = ["train", "test", "dev"]
# models = ["gemma2:27b", "llama3.1:70b"]
# seeds = [0, 1, 2, 3, 4]
# modes = ["chain-of-thought", "plan-and-solve", "label"] # "label"

seeds = [0]
n_few_shot = [10, 50] # 0 fehlt noch
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera", "gerest"]
tasks = ["asqp", "tasd"]
models = ["gemma2:27b"]

n_synth = 2000


combinations = itertools.product(seeds, n_few_shot, datasets, tasks, models)

for combination in combinations:
    seed, fs,  dataset_name, task, model = combination
    file_path = f"generations/augmentations/{task}_{dataset_name}_{model}_{seed}_{fs}.json"
    # Prüfen, ob die Datei bereits existiert
    if not os.path.exists(file_path):
        create_annotations(task, dataset_name, model, seed, fs, n_synth)
    else:
        print(f"Skipping: {file_path} already exists.")