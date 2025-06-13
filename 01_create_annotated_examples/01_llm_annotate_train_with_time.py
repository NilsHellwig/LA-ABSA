import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../zero-shot-absa-quad/')))

from validator import validate_label, validate_reasoning
from similarity import sort_examples_by_similarity
from promptloader import PromptLoader
from dataloader import DataLoader
from llm import LLM
import itertools
import json
import random, time, subprocess, threading


vram_values = []
watt_values = []

running = False

def monitor_gpu(interval=0.1):
    global running
    global vram_values 
    global watt_values
    
    vram_values = []
    watt_values = []
    while running:
        try:
            # Ausgabe von nvidia-smi
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used,power.draw", "--format=csv,noheader,nounits"],
                encoding='utf-8'
            )
            lines = result.strip().split('\n')
            for line in lines:
                mem_str, watt_str = line.split(',')
                vram = float(mem_str.strip())     # in MiB
                watt = float(watt_str.strip())    # in Watt
                vram_values.append(vram)
                watt_values.append(watt)
        except Exception as e:
            print(f"Fehler beim Auslesen von nvidia-smi: {e}")
        time.sleep(interval)


## Load API Key
from dotenv import load_dotenv
load_dotenv()
GWDG_KEY = os.getenv("GWDG_KEY")  

## LLM

dataloader = DataLoader(base_path="../zero-shot-absa-quad/datasets/")
promptloader = PromptLoader(base_path="../zero-shot-absa-quad/prompt/")

SPLIT_SEED = 42

import concurrent.futures

def run_with_timeout(llm, prompt, seed, temp, timeout=300):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llm.predict, prompt, seed, stop=["]"], temperature=temp)
        try:
            output, duration = future.result(timeout=timeout)
            return output, duration
        except concurrent.futures.TimeoutError:
            raise TimeoutError("Vorhersage hat länger als 20 Sekunden gedauert.")



def create_annotations(TASK, DATASET_NAME, DATASET_TYPE, LLM_BASE_MODEL, SEED, N_FEW_SHOT, SORT_EXAMPLES):
    
    print(f"{TASK}_{DATASET_NAME}_{DATASET_TYPE}_{LLM_BASE_MODEL}_{SEED}_{N_FEW_SHOT}.json")

    print(f"TASK:", TASK)
    print(f"DATASET_NAME: {DATASET_NAME}")
    print(f"DATASET_TYPE: {DATASET_TYPE}")
    print(f"LLM_BASE_MODEL: {LLM_BASE_MODEL}")
    print(f"SEED: {SEED}")
    print(f"N_FEW_SHOT: {N_FEW_SHOT}")
    print(f"SORT_EXAMPLES: {SORT_EXAMPLES}")

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
    
    ## Load Few-Shot Dataset
    few_shot_split_0 = []

    dataset_train = dataloader.load_data(name=DATASET_NAME, data_type="train", target=TASK)
    few_shot_split_0 = dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[0] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[1] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[2] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[3] + dataloader.random_cross_validation_split(dataset_train, seed=SPLIT_SEED)[4]
    random.seed(SPLIT_SEED)
    dataset_annotation = few_shot_split_0[N_FEW_SHOT:]

    if (N_FEW_SHOT > 0):
        few_shot_split_0 = few_shot_split_0[0:N_FEW_SHOT]
        print(len(few_shot_split_0), len(dataset_annotation), len(dataset_train))
    else:
        few_shot_split_0 = []
          
    fs_examples_ids = [int(example["id"].split("_")[0]) for example in few_shot_split_0]

    # Lade alle Zeilen aus der Datei
    fs_examples_txt = ""
    with open(f"../zero-shot-absa-quad/datasets/{TASK}/{DATASET_NAME}/train.txt", "r") as f:
        lines = f.readlines()

        # Füge die Zeilen zusammen, deren Index in fs_examples_ids enthalten ist
        fs_examples_txt = "".join(lines[i] for i in fs_examples_ids if 0 <= i < len(lines))

    # Erstelle den Zielpfad, falls er nicht existiert
    output_dir_fs = f"../zero-shot-absa-quad/fs_examples/{TASK}/{DATASET_NAME}/fs_{N_FEW_SHOT}"
    os.makedirs(output_dir_fs, exist_ok=True)

    # Speichere die resultierenden Beispiele in einer Datei
    output_path_fs = os.path.join(output_dir_fs, "examples.txt")
    with open(output_path_fs, "w") as f:
        f.write(fs_examples_txt)
        
    #############################################
    #############################################

    ## label
    dataset_annotation = random.sample(dataset_annotation, int(len(dataset_annotation) * 0.05))
    for idx, example in enumerate(dataset_annotation):
        prediction = { 
            "task": TASK,
            "dataset_name": DATASET_NAME, 
            "dataset_type": DATASET_TYPE,
            "llm_base_model": LLM_BASE_MODEL,
            "id": example["id"], 
            "invalid_precitions_label": [],
            "init_seed": SEED,
        }
        
        seed = SEED
        
        if SORT_EXAMPLES == True:
           few_shot_split_0 = sort_examples_by_similarity(example, few_shot_split_0)
    
        prompt = promptloader.load_prompt(task=TASK,
                                      prediction_type="label", 
                                      aspects=unique_aspect_categories, 
                                      examples=few_shot_split_0,
                                      seed_examples=seed,
                                      input_example=example, shuffle_examples=SORT_EXAMPLES==False)
    
   
        global running 
        running = True
        start_time = time.time()
        monitor_thread = threading.Thread(target=monitor_gpu)
        monitor_thread.start()
            
        correct_output = False   
        while correct_output == False:
            generated = False
            k = 0
            while generated == False:
                try:
                    print("Start")
                    output, duration = run_with_timeout(llm, prompt, seed, temp=0.8+k)
                    generated = True
                except Exception as e:
                    print("Exception")
                    #k += 0.01
                    pass
                
            output_raw = output
            
            # delete new lines
            output = output.replace("\n", "")
            
            validator_output = validate_label(output, example["text"], unique_aspect_categories, task=TASK, allow_small_variations=True)

            if validator_output[0] != False:
                prediction["pred_raw"] = output_raw
                prediction["pred_label"] = validator_output[0]
                prediction["duration_label"] = duration
                prediction["seed"] = seed
                correct_output = True
            else:
                prediction["invalid_precitions_label"].append({"pred_label_raw": output_raw, "pred_label": validator_output[0], "duration_label": duration, "seed": seed, "regeneration_reason": validator_output[1]})
                seed += 5
                pass
        
            if len(prediction["invalid_precitions_label"]) > 9:
                correct_output = True
                prediction["pred_label"] = []
                prediction["duration_label"] = duration
                prediction["seed"] = seed
 
        running = False
        monitor_thread.join()
    
        mean_vram = sum(vram_values) / len(vram_values) if vram_values else 0
        mean_watt = sum(watt_values) / len(watt_values) if watt_values else 0
        
        prediction["avg_gpu_power_eval_W"] = mean_watt
        prediction["avg_gpu_vram_eval_MB"] = mean_vram
        prediction["total_time_eval"] = time.time() - start_time
    
        print(f"####### {idx}/{len(dataset_annotation)} ### ", idx, "\nText:", example["text"], "\nLabel:",prediction["pred_label"], "\nRegenerations:", prediction["invalid_precitions_label"])
        predictions.append(dict(prediction, **example))

    dir_path = f"_out_synthetic_examples/01_llm_annotate_train"

    # Create the directories if they don't exist
    os.makedirs(dir_path, exist_ok=True)

    with open(f"{dir_path}/{TASK}_{DATASET_NAME}_{DATASET_TYPE}_{LLM_BASE_MODEL}_{SEED}_{N_FEW_SHOT}_PERFORMANCE.json", 'w', encoding='utf-8') as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)
        
        

##### create annotations

# tasks = ["asqp", "tasd"]
# datasets = ["rest15", "rest16"]
# dataset_types = ["train", "test", "dev"]
# models = ["gemma3:27b", "llama3.1:70b"]
# seeds = [0, 1, 2, 3, 4]

seeds = [0, 1, 2, 3, 4]
n_few_shot = [50, 10, 0] # 0 fehlt noch
datasets = ["rest15", "rest16", "hotels", "flightabsa", "coursera"]
tasks = ["asqp", "tasd"]
dataset_types = ["train"]
models = ["gemma3:27b"]
sort_examples = [False]


combinations = itertools.product(seeds, n_few_shot, datasets, tasks, dataset_types, models, sort_examples)

for combination in combinations:
    seed, fs,  dataset_name, task, dataset_type, model, s_ex = combination
    file_path = f"_out_synthetic_examples/01_llm_annotate_train/{task}_{dataset_name}_{dataset_type}_{model}_{seed}_{fs}_PERFORMANCE.json"
    # Prüfen, ob die Datei bereits existiert
    if not os.path.exists(file_path):
        # time.sleep(3)
        # subprocess.run(["ollama", "stop", model])
        create_annotations(task, dataset_name, dataset_type, model, seed, fs, s_ex)
    else:
        print(f"Skipping: {file_path} already exists.")