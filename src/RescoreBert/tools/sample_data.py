import json
import random
from pathlib import Path

random.seed(42)
dataset = "aishell"
setting = "noLM"
tasks = ["train", "dev"]
for task in tasks:
    with open(f"../../../data/{dataset}/data/{setting}/{task}/data.json") as src:
        data_json = json.load(src)

    sample_data = []
    for data in data_json:
        correct_flag = False  # contain correct flag

        random_sample = random.sample(data["hyps"], 5)
        labels = []
        for hyp in random_sample:
            sample_data.append(
                {
                    "hyp": hyp,
                    "ref": data["ref"],
                    "label": 1 if hyp == data["ref"] else 0,
                }
            )

        sample_data.append({"hyp": data["ref"], "ref": data["ref"], "label": 1})

    train_path = Path(f"../data/{dataset}/{setting}/{task}")
    train_path.mkdir(parents=True, exist_ok=True)
    with open(f"{train_path}/pretrain_data.json", "w") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=1)
