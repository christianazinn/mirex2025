from pathlib import Path
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

        # RWKV requires inputs to have a shape divisible by 16. Normally, this
        # would be done by the data collator but since we are using the miditok data collator,
        # which doesn't support pad_to_mulitple_of, we do it here instead.
        shape_remainder = input_ids.shape[-1] % 16
        input_ids = torch.nn.functional.pad(
            input_ids, [0, (16 - shape_remainder)], value=self._tokenizer.pad_token_id
        )
        attention_mask = torch.nn.functional.pad(
            attention_mask,
            [0, (16 - shape_remainder)],
            value=self._tokenizer.pad_token_id,
        )
        labels = torch.nn.functional.pad(
            labels, [0, (16 - shape_remainder)], value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
