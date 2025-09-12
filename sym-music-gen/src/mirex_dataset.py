from pathlib import Path
import numpy as np
import torch
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
        self._num_completion_measures = 12
        self._max_pitch_offset = max_pitch_offset

    def __len__(self):
        return len(self._midi_files)

    def __getitem__(self, idx: int):
        file_path = self._midi_files[idx]

        score = symusic.Score(file_path)
        encoding = self._tokenizer.encode(score)[0]

        token_ids = np.array(encoding.ids)
        bar_starts = np.where(token_ids == self._tokenizer.vocab["Bar_None"])[0]
        for _ in range(10):
            selected_bar_start = np.random.randint(0, len(bar_starts) - 16)
            sample_start = bar_starts[selected_bar_start]
            sample_end = bar_starts[selected_bar_start + 16]

            sample = token_ids[sample_start:sample_end]

            if len(sample) > 100:
                break

        input_ids = torch.from_numpy(sample)
        attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype)

        # labels must be shifted
        labels = input_ids.clone()[1:]
        input_ids = input_ids[:-1]

        # truncate both to 1024 (hardcoded for now)
        tgt_len = min(len(labels), len(input_ids), 1024)
        labels = labels[:tgt_len]
        input_ids = input_ids[:tgt_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
