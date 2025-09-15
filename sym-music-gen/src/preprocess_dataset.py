import argparse
import json
import os
import miditok
import datasets
from pathlib import Path


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-proc", type=int, default=os.cpu_count() - 1)

    return parser.parse_args()


def main():
    args = __parse_args()

    files = [str(x.resolve()) for x in args.data.glob("**/*.mid")]

    dataset = datasets.Dataset.from_dict({"path": files})

    config = miditok.TokenizerConfig(
        pitch_range=(0, 127),
        use_velocities=False,
        encode_ids_splits="no",
        use_pitchdrum_tokens=False,
    )

    tokenizer = miditok.REMI(config)

    def tokenize(sample):
        enc = tokenizer.encode(sample["path"])

        tokens = enc[0].ids + [tokenizer.vocab["PAD_None"]]

        bar_starts = [
            i for i, v in enumerate(tokens) if v == tokenizer.vocab["Bar_None"]
        ]

        return {"tokens": tokens, "bar_start": bar_starts}

    pitch_tokens = [v for k, v in tokenizer.vocab.items() if k.startswith("Pitch")]

    dataset = dataset.map(
        tokenize,
        remove_columns=["path"],
        num_proc=args.num_proc,
        desc="Tokenizing",
    )

    dataset.save_to_disk(args.output)

    with (args.output / "metadata.json").open("w") as io:
        json.dump({"pitch_tokens": pitch_tokens}, io)


if __name__ == "__main__":
    main()
