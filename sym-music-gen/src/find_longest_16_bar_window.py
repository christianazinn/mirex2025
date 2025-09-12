from pathlib import Path
import os
import concurrent.futures as cf

import numpy as np

from tqdm import tqdm

import symusic, miditok

# Global in workers (inherited via fork on Linux)
TOKENIZER = None
BAR_IDS = None

def set_tokenizer(tok):
    """Call this once in the main process before parallel()"""
    global TOKENIZER, BAR_IDS
    TOKENIZER = tok
    # Accept both "Bar_None" and any "Bar_*" variants
    BAR_IDS = np.array([i for t, i in tok.vocab.items() if t.startswith("Bar")], dtype=np.int32)
    if BAR_IDS.size == 0:
        raise RuntimeError("No Bar tokens found in tokenizer.vocab")

def _file_max_len_16bars(midi_path: str) -> int:
    try:
        score = symusic.Score(midi_path)
        ids = np.asarray(TOKENIZER.encode(score)[0].ids, dtype=np.int32)

        # bar starts (vectorized)
        bar_starts = np.flatnonzero(np.isin(ids, BAR_IDS))
        if bar_starts.size < 17:
            return 0

        # length of every exact 16-bar window: start[i+16] - start[i]
        lens = bar_starts[16:] - bar_starts[:-16]
        return int(lens.max()) if lens.size else 0

    except Exception:
        return 0

def parallel_longest_16bar_window(midi_root, tokenizer, max_workers=None, chunksize=8):
    set_tokenizer(tokenizer)  # inherit to children via fork
    midi_files = (
        [str(p) for p in Path(midi_root).glob("**/*.mid")]
        + [str(p) for p in Path(midi_root).glob("**/*.midi")]
    )
    if not midi_files:
        return 0

    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)

    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        it = ex.map(_file_max_len_16bars, midi_files, chunksize=chunksize)
        return max(tqdm(it, total=len(midi_files), desc="Scanning"), default=0)

if __name__ == '__main__':
    config = miditok.TokenizerConfig(
        pitch_range=(0, 127),
        use_velocities=False,
        encode_ids_splits="no",
        use_pitchdrum_tokens=False,
    )
    tokenizer = miditok.REMI(config)

    midi_files = Path('../data/filtered_aria/train')
    print(parallel_longest_16bar_window(midi_files, tokenizer))
    
    # Result: 2695
