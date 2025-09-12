import json
import tqdm
import shutil
import random
from pathlib import Path

random.seed(1337)

dataset_path = Path("/home/christian/mirex2025/aria-midi-v1-pruned-ext")
frac_val = 0.1
min_score = 0.9


output_path = Path.cwd() / "data" / "filtered_aria"
output_path.mkdir()
(output_path / "train").mkdir()
(output_path / "val").mkdir()

with (dataset_path / "metadata.json").open() as io:
    metadata = json.load(io)

print("Metadata loaded")

split = {}

for sample_path in tqdm.tqdm((dataset_path / "data").glob("**/*.mid")):
    # For each sample randomly put it into the train or validation sets
    sample_idx, score_idx = sample_path.stem.split("_")

    is_train = split.get(sample_idx)

    if is_train is None:
        is_train = random.random() > frac_val
        split[sample_idx] = is_train

    if str(sample_idx) not in metadata:
        print(f"Sample {sample_idx} not in metadata")
        continue

    if metadata[str(sample_idx)]["audio_scores"][str(score_idx)] > min_score:
        if is_train:
            dst_path = output_path / "train" / f"{sample_idx}_{score_idx}.mid"
        else:
            dst_path = output_path / "val" / f"{sample_idx}_{score_idx}.mid"

        shutil.copy(sample_path, dst_path)
