import argparse
import json
from tqdm import tqdm
import os
import miditok
import datasets
from pathlib import Path


def __parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--num-proc", type=int, default=os.cpu_count() - 1)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--min-avg-notes-per-bar", type=float, default=5.0)
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def main():
    args = __parse_args()
    files = [str(x.resolve()) for x in args.data.glob("**/*.mid")][0:100]
    dataset = datasets.Dataset.from_dict({"path": files})
    config = miditok.TokenizerConfig(
        pitch_range=(0, 127),
        use_velocities=False,
        encode_ids_splits="no",
        use_pitchdrum_tokens=False,
    )
    pad_token_id = 0

    def pad_or_truncate(example):
        tokens = example["tokens"][: args.max_length]
        if len(tokens) < args.max_length:
            tokens = tokens + [pad_token_id] * (args.max_length - len(tokens))

        bar_starts = [x for x in example["bar_starts"] if x < args.max_length]
        example["tokens"] = tokens
        example["bar_starts"] = bar_starts
        return example

    def tokenize(sample):
        tokenizer = miditok.REMI(config)
        enc = tokenizer.encode(sample["path"])
        tokens = enc[0].ids
        bar_starts = [
            i for i, v in enumerate(tokens) if v == tokenizer.vocab["Bar_None"]
        ]
        pitch_tokens = {v for k, v in tokenizer.vocab.items() if k.startswith("Pitch")}
        num_bars = len(bar_starts)
        if num_bars < args.window_size:
            return {"tokens": []}
        step = args.window_size - args.overlap
        expanded_tokens = []
        expanded_bar_starts = []
        for start_bar in range(0, num_bars - args.window_size + 1, step):
            if len(expanded_tokens) > 100:
                break
            note_counts = []
            local_bar_starts = []
            for b in range(start_bar, start_bar + args.window_size):
                bar_start_idx = bar_starts[b]
                bar_end_idx = bar_starts[b + 1] if b + 1 < num_bars else len(tokens)
                bar_tokens = tokens[bar_start_idx + 1 : bar_end_idx]
                notes = sum(t in pitch_tokens for t in bar_tokens)
                note_counts.append(notes)
                local_bar_starts.append(bar_start_idx - bar_starts[start_bar])
            avg_notes = sum(note_counts) / args.window_size
            if avg_notes < args.min_avg_notes_per_bar:
                continue
            window_start_idx = bar_starts[start_bar]
            window_end_idx = (
                bar_starts[start_bar + args.window_size]
                if start_bar + args.window_size < num_bars
                else len(tokens)
            )
            window_tokens = tokens[window_start_idx:window_end_idx]
            expanded_tokens.append(window_tokens)
            expanded_bar_starts.append(local_bar_starts)
        return {"tokens": expanded_tokens, "bar_starts": expanded_bar_starts}

    dataset = dataset.map(
        tokenize,
        num_proc=args.num_proc,
        desc="Tokenizing and expanding",
    )
    dataset = dataset.filter(
        lambda x: len(x["tokens"]) > 0 and isinstance(x["tokens"], list), num_proc=8
    )

    with open("intermediate.jsonl", "w") as intermediate_io:
        for idx in tqdm(range(len(dataset))):
            sample = dataset[idx]
            for tok, bar_starts in zip(sample["tokens"], sample["bar_starts"]):
                new = {"tokens": tok, "bar_starts": bar_starts}
                intermediate_io.write(f"{json.dumps(new)}\n")

    dataset = datasets.load_dataset("json", data_files="intermediate.jsonl")
    dataset = dataset.map(pad_or_truncate, num_proc=8)
    tokenizer = miditok.REMI(config)
    pitch_tokens = [v for k, v in tokenizer.vocab.items() if k.startswith("Pitch")]
    dataset.save_to_disk(args.output)
    with (args.output / "metadata.json").open("w") as io:
        json.dump({"pitch_tokens": pitch_tokens}, io)


if __name__ == "__main__":
    main()
