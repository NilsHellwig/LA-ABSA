from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import shutil

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader import DataLoader
from promptloader import PromptLoader
from datasets import Dataset

promptloader = PromptLoader()
dataloader = DataLoader("./datasets", "./fs_examples")


def fine_tune_llm(seed, ds_name, fs_num, task, n_llm_examples):
         train_ds = dataloader.load_data(ds_name, "train", cv=False, target=task, fs_num=fs_num, fs_ann_mode=True, n_ann_examples=n_llm_examples)[fs_num:]
         fs_ds = dataloader.load_data(ds_name, "train", cv=False, target=task, fs_num=fs_num, fs_mode=True)
         test_ds = dataloader.load_data(ds_name, "test", cv=False, target=task)
    
         unique_aspect_categories = sorted({aspect['aspect_category'] for entry in dataloader.load_data(name=ds_name, data_type="all", target=task) for aspect in entry['aspects']})
         
         dataset = []
    
         for idx, example in enumerate(train_ds):
             prompt = promptloader.load_prompt(task=task,
                                      prediction_type="label", 
                                      aspects=unique_aspect_categories, 
                                      examples=fs_ds,
                                      seed_examples=seed,
                                      input_example=example)
             dataset.append({"text": example["text"], "output": str(example["tuple_list"]), "instruction": "Do Aspect Sentiment Quadruple Preiction."})
             
             
         # Dataset aus der Liste von Dicts erstellen
         dataset = Dataset.from_dict({key: [d[key] for d in dataset] for key in dataset[0]})
      

         max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
         dtype = (
             None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
         )
         load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

         model, tokenizer = FastLanguageModel.from_pretrained(
             model_name="unsloth/gemma-2-9b",
             max_seq_length=max_seq_length,
             dtype=dtype,
             load_in_4bit=load_in_4bit,
             # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
         )

         model = FastLanguageModel.get_peft_model(
             model,
             r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
             target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj",],
             lora_alpha = 16,
             lora_dropout = 0, # Supports any, but = 0 is optimized
             bias = "none",    # Supports any, but = "none" is optimized
             # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
             use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
             random_state = 3407,
             use_rslora = False,  # We support rank stabilized LoRA
             loftq_config = None, # And LoftQ
         )

         alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

         ### Instruction:
         {}

         ### Input:
         {}

         ### Response:
         {}"""

         EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
         def formatting_prompts_func(examples):
             instructions = examples["instruction"]
             inputs       = examples["input"]
             outputs      = examples["output"]
             texts = []
             for instruction, input, output in zip(instructions, inputs, outputs):
                 # Must add EOS_TOKEN, otherwise your generation will go on forever!
                 text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
                 texts.append(text)
             return { "text" : texts, }

 
         #  dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
         #  dataset = dataset.map(formatting_prompts_func, batched = True,)
         


         trainer = SFTTrainer(
             model = model,
             tokenizer = tokenizer,
             train_dataset = dataset,
             dataset_text_field = "text",
             max_seq_length = max_seq_length,
             dataset_num_proc = 2,
             packing = False, # Can make training 5x faster for short sequences.
             args = TrainingArguments(
                 per_device_train_batch_size = 2,
                 gradient_accumulation_steps = 4,
                 warmup_steps = 5,
                 max_steps = 250,
                 learning_rate = 2e-4,
                 fp16 = not is_bfloat16_supported(),
                 bf16 = is_bfloat16_supported(),
                 logging_steps = 1,
                 optim = "adamw_8bit",
                 weight_decay = 0.01,
                 lr_scheduler_type = "linear",
                 seed = 3407,
                 output_dir = "outputs",
                 report_to = "none", # Use this for WandB etc
             ),
         )

         trainer_stats = trainer.train()

         FastLanguageModel.for_inference(model) # Enable native 2x faster inference
         inputs = tokenizer(
         [
             "Das Essen war lecker aber der Service war schrecklich.",
         ], return_tensors = "pt").to("cuda")

         outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
         print(tokenizer.batch_decode(outputs))
         print("------")


         shutil.rmtree('outputs')
         
         
for i in range(5):
 for ds_name in ["rest16", "hotels", "rest15", "flightabsa", "coursera"]:
  for fs_num in [50, 10, 0]:
   for task in ["asqp", "tasd"]:
     for n_llm_examples in ["full", 500, 800]:  
         fine_tune_llm(seed=i, ds_name=ds_name, fs_num=fs_num, task=task, n_llm_examples=n_llm_examples)