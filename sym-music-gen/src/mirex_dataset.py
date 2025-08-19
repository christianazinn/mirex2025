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
        self._num_completion_measures = 12
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

        # 1. Take first four bars of the sample
        # 2. Search for similar music to those four bars in the database

        # 3. Return an augmented sample including either those entire
        #    pieces, or just the 12-bar continuation, in the format

        # The goal is to, given the 4-bar prompt P,

        # 1. Search for snippets similar to the prompt in the [Aria
        #    dataset](https://arxiv.org/abs/2504.15071), S1, ..., SN with
        #    respective 12-bar continuations C1, ..., CN. This is the
        #    point where the music database is necessary, to identify the
        #    similar snippets.
        
        # 2. Provide the following augmented prompt to an LLM, which
        #    will be trained on such prompts P with continuation C taken
        #    from the Aria dataset:

        #    S1ǁC1[RAG_SEP] ... [RAG_SEP]SNǁCN[PROMPT_START]P[GEN_START]C

        #    (Here the continuation "C" is only included during training,
        #    and next-token prediction is only done on "C". [RAG_SEP],
        #    [PROMPT_START] and [GEN_START] are special tokens. During
        #    inference, the sequence is
             
        #    S1ǁC1[RAG_SEP] ... [RAG_SEP]SNǁCN[PROMPT_START]P[GEN_START]
             
        #    and C is the product of the inference.)

        selected_bar_start = np.random.randint(0, len(bar_starts) - 16)
        sample_start = bar_starts[selected_bar_start]
        sample_end = bar_starts[selected_bar_start + 16]

        sample = token_ids[sample_start:sample_end]

        input_ids = torch.from_numpy(sample)
        attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype)

        # labels must be shifted
        labels = input_ids.clone()[1:]
        input_ids = input_ids[:-1]

        # truncate both to 1024 (hardcoded for now)
        tgt_len = min(len(labels), len(input_ids), 1024)
        labels = labels[:tgt_len]
        input_ids = input_ids[:tgt_len]

        # Get the start of the completion measures
        # index_start_completion = torch.where(
        #     labels == self._tokenizer.vocab["Bar_None"]
        # )[0][self._num_prompt_measures].item()

        # labels[0:index_start_completion] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
