import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
import shutil
from validator import validate_label

import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import DataLoader
from promptloader import PromptLoader
from datasets import Dataset

import json
import torch

promptloader = PromptLoader()
dataloader = DataLoader("./datasets", "./fs_examples")

alpaca_prompt_base = """

         ### Instruction:
         {}

         ### Input:
         {}

         ### Response:
         {}"""


def get_trainer(model, tokenizer, dataset, max_seq_length):
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_steps=0,
            max_steps=int(len(dataset) / 32) * 10, # 8 is the batch size
            learning_rate=3e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=43,
            output_dir="outputs",
            report_to="none",  # Use this for WandB etc
        ),
    )


def get_model_and_tokenizer(max_seq_length):
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=f"unsloth/{LLM_NAME}",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=8,
        lora_dropout=0.05,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=43,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    return model, tokenizer


def fine_tune_llm(seed, ds_name, fs_num, task, n_llm_examples, llm_name):
    # Load datasets
    train_ds = dataloader.load_data(
        ds_name,
        "train",
        cv=False,
        target=task,
        fs_num=fs_num,
        fs_ann_mode=True,
        n_ann_examples=n_llm_examples,
    )[fs_num:]
    unique_aspect_categories = sorted(
        {
            aspect["aspect_category"]
            for entry in dataloader.load_data(
                name=ds_name, data_type="all", target=task
            )
            for aspect in entry["aspects"]
        }
    )

    prompt_header = promptloader.load_prompt(task=task, load_llm_instruction=True)
    test_ds = dataloader.load_data(ds_name, "test", cv=False, target=task)

    # Create prompt from dataset
    alpaca_prompt = alpaca_prompt_base
    dataset = []
    for idx, example in enumerate(train_ds):
        dataset.append(
            {
                "text": example["text"],
                "output": str(example["tuple_list"]),
                "instruction": prompt_header,
                "input": example["text"],
            }
        )
    dataset = Dataset.from_dict({key: [d[key] for d in dataset] for key in dataset[0]})
    max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
    model, tokenizer = get_model_and_tokenizer(max_seq_length)
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {
            "text": texts,
        }

    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    # Run training
    trainer = get_trainer(model, tokenizer, dataset, max_seq_length)
    trainer.train()

    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    predictions = []
    for idx, example in enumerate(test_ds):

        prediction = {
            "task": task,
            "dataset_name": ds_name,
            "dataset_type": "test",
            "llm_base_model": llm_name,
            "mode": "label",
            "id": example["id"],
            "invalid_precitions_label": [],
            "init_seed": seed,
        }

        correct_output = False
        while correct_output == False:

            test_text = example["text"]

            inputs = tokenizer(
                [
                    alpaca_prompt.format(
                        prompt_header,  # instruction
                        test_text,  # input
                        "",  # output - leave this blank for generation!
                    )
                ],
                return_tensors="pt",
            ).to("cuda")
            
            torch.manual_seed(len(prediction["invalid_precitions_label"]))

            output_raw = tokenizer.batch_decode(
                model.generate(**inputs, max_new_tokens=64, use_cache=True),
            )[0]
            output_label = output_raw.split("### Response:")

            if len(output_label) < 2:
                continue

            output_label = output_label[1].strip()
            validator_output = validate_label(
                output_label, test_text, unique_aspect_categories, task=task
            )

            if validator_output[0] != False:
                prediction["pred_raw"] = output_raw
                prediction["pred_label"] = validator_output[0]
                prediction["seed"] = seed
                correct_output = True
            else:
                prediction["invalid_precitions_label"].append(
                    {
                        "pred_label_raw": output_raw,
                        "pred_label": validator_output[0],
                        "seed": seed,
                        "regeneration_reason": validator_output[1],
                    }
                )
                seed += 5
                pass

            if len(prediction["invalid_precitions_label"]) > 9:
                correct_output = True
                prediction["pred_label"] = []
                prediction["seed"] = seed

        print(
            f"####### {idx}/{len(test_ds)} ### ",
            "\nText:",
            example["text"],
            "\nLabel:",
            prediction["pred_label"],
            "\nRegenerations:",
            len(prediction["invalid_precitions_label"]),
        )
        predictions.append(dict(prediction, **example))

    dir_path = f"generations/ft_llm_predictions"

    # Create the directories if they don't exist
    os.makedirs(dir_path, exist_ok=True)

    with open(
        f"{dir_path}/fine_tune_{llm_name}_{seed}_{task}_{fs_num}_{ds_name}.json",
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(predictions, json_file, ensure_ascii=False, indent=4)

    shutil.rmtree("outputs")


LLM_NAME = "gemma-2-9b"

for i in range(5):
    for ds_name in ["rest16", "hotels", "rest15", "flightabsa", "coursera"]:
        for fs_num in [50, 10, 0]:
            for task in ["asqp", "tasd"]:
                for n_llm_examples in ["full", 500, 800]:
                    fine_tune_llm(
                        seed=i,
                        ds_name=ds_name,
                        fs_num=fs_num,
                        task=task,
                        n_llm_examples=n_llm_examples,
                        llm_name=LLM_NAME
                    )
