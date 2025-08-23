from pathlib import Path
import json
import datasets
import numpy as np
import torch
import miditok
import symusic


class MIREXCustomDataset(torch.utils.data.Dataset):
    def __init__(
        self, midi_files: Path | list[Path], tokenizer, max_pitch_offset: int = 0
    ):
        super().__init__()
        self._midi_files = (
            midi_files
            if isinstance(midi_files, list)
            else list(midi_files.glob("**/*.mid"))
        )

        self._tokenizer = tokenizer

        self._num_prompt_measures = 4
        self._num_completion_measures = 4
        self._max_pitch_offset = max_pitch_offset

    def __len__(self):
        return len(self._midi_files)

    def __getitem__(self, idx: int):
        file_path = self._midi_files[idx]

        score = symusic.Score(file_path)

        if self._max_pitch_offset > 0:
            # Get max and min pitch to make sure the augmentation method doesn't crash
            max_pitch = max(x.pitch for x in score.tracks[0].notes)
            min_pitch = min(x.pitch for x in score.tracks[0].notes)
            min_augment_pitch = max(0, min_pitch - self._max_pitch_offset) - min_pitch
            max_augment_pitch = min(127, max_pitch + self._max_pitch_offset) - max_pitch
            pitch_augmentation = np.random.randint(min_augment_pitch, max_augment_pitch)

            score = miditok.data_augmentation.augment_score(
                score, pitch_offset=pitch_augmentation
            )

        encoding = self._tokenizer.encode(score)[0]

        token_ids = np.array(encoding.ids)

        # Select random bar to start
        bar_starts = np.where(token_ids == self._tokenizer.vocab["Bar_None"])[0]

        selected_bar_start = np.random.randint(0, len(bar_starts) - 16)
        sample_start = bar_starts[selected_bar_start]
        sample_end = bar_starts[selected_bar_start + 16]

        sample = token_ids[sample_start:sample_end]

        input_ids = torch.from_numpy(sample)
        attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype)

        labels = input_ids.clone()

        # Get the start of the completion measures
        index_start_completion = torch.where(
            labels == self._tokenizer.vocab["Bar_None"]
        )[0][self._num_prompt_measures].item()

        labels[0:index_start_completion] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MIREXPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path, max_pitch_offset: int = 0):
        super().__init__()
        self._ds = datasets.load_from_disk(dataset_path)

        if "train" in self._ds:
            self._ds = self._ds["train"]

        self._ds.set_format("torch")

        with (dataset_path / "metadata.json").open() as io:
            self._pitch_tokens = torch.tensor(json.load(io)["pitch_tokens"])
            self._pitch_token_offset = self._pitch_tokens[0]

        self._max_pitch_offset = max_pitch_offset
        self._num_prompt_bars = 4

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx: int):
        sample = self._ds[idx]

        input_ids = torch.as_tensor(sample["tokens"], dtype=torch.long)[0:1024]
        bar_starts = [x for x in sample["bar_starts"] if x < 1024]

        if self._max_pitch_offset > 0:
            pitch_tokens = torch.isin(input_ids, self._pitch_tokens)
            max_pitch = input_ids[pitch_tokens].max() - self._pitch_token_offset
            min_pitch = input_ids[pitch_tokens].min() - self._pitch_token_offset

            min_augment_pitch = (
                max(0, min_pitch.item() - self._max_pitch_offset) - min_pitch.item()
            )
            max_augment_pitch = (
                min(127, max_pitch.item() + self._max_pitch_offset) - max_pitch.item()
            )

            augment_pitch = torch.randint(min_augment_pitch, max_augment_pitch, (1,))

            input_ids[pitch_tokens] += augment_pitch

        attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype)
        attention_mask[input_ids == 0] = 0  # Don't attend to pad tokens
        labels = input_ids.clone()

        prompt_bar_end = bar_starts[self._num_prompt_bars]
        # labels[0:prompt_bar_end] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
