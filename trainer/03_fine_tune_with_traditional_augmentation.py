import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataloader import DataLoader
from trainer import train_paraphrase, train_mvp, train_dlo
import json
import time

from helper import clean_up, create_output_directory

dataloader = DataLoader("./datasets", "./fs_examples")


for i in range(5):
    for ds_name in ["rest16", "hotels", "rest15", "flightabsa", "coursera"]:
        for fs_num in [10, 50]:
            for task in ["asqp", "tasd"]:
                for n_llm_examples in ["full", 800, 1600]:
                    for aug_method in ["eda", "llm_eda", "back_translation"]:
                        train_ds = dataloader.load_data(
                            ds_name,
                            "train",
                            cv=False,
                            target=task,
                            fs_num=fs_num,
                            aug_mode=True,
                            aug_method=aug_method,
                            n_ann_examples=n_llm_examples,
                        )

                        test_ds = dataloader.load_data(
                            ds_name, "test", cv=False, target=task
                        )

                        for ml_method in ["paraphrase", "dlo"]:
                            print(
                                f"Task:",
                                task,
                                "Dataset:",
                                ds_name,
                                "Seed:",
                                i,
                                "ML-Method:",
                                ml_method,
                                "FS-Num:",
                                fs_num,
                                "len(train_ds)",
                                len(train_ds),
                                "len(test_ds)",
                                len(test_ds),
                                "n_llm_examples",
                                n_llm_examples,
                                "aug_method",
                                aug_method,
                            )
                            filename = f"./generations/train_traditional_augmentations/training_{task}_{ds_name}_seed-{i}_n-train_{ml_method}_fs-num_{fs_num}_n-llm-examples_{n_llm_examples}_aug_method_{aug_method}.json"

                            if os.path.exists(filename):
                                print(f"File {filename} already exists. Skipping.")
                                continue
                            else:

                                clean_up()
                                create_output_directory()

                                if ml_method == "paraphrase":
                                    scores = train_paraphrase(
                                        train_ds=train_ds,
                                        test_ds=test_ds,
                                        seed=i,
                                        dataset=ds_name,
                                        task=task,
                                    )
                                if ml_method == "mvp":
                                    scores = train_mvp(
                                        train_ds=train_ds,
                                        test_ds=test_ds,
                                        seed=i,
                                        dataset=ds_name,
                                        task=task,
                                    )
                                if ml_method == "dlo":
                                    scores = train_dlo(
                                        train_ds=train_ds,
                                        test_ds=test_ds,
                                        seed=i,
                                        dataset=ds_name,
                                        task=task,
                                    )

                                with open(filename, "w", encoding="utf-8") as json_file:
                                    json.dump(
                                        scores, json_file, ensure_ascii=False, indent=4
                                    )


