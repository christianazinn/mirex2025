import math
import torch
from torch.nn.utils.rnn import pad_sequence


def make_collate_fn(
    pad_id: int,
    fixed_len: int | None = None,
    return_mask: bool = False,
    chunk_multiple: int = 16,
):
    """
    Pads sequences in a batch so they are all the same length,
    rounding up to the nearest multiple of `chunk_multiple`.

    Labels are padded with -100 so they are ignored by cross-entropy loss.
    Attention masks are padded with 0.
    """

    def _collate(batch):
        input_ids = [b["input_ids"] for b in batch]
        labels = [b["labels"] for b in batch]
        attn = [b["attention_mask"] for b in batch]

        # Determine base target length
        T_raw = max(x.size(0) for x in input_ids) if fixed_len is None else int(fixed_len)

        # Round up to next multiple of `chunk_multiple`
        if chunk_multiple and chunk_multiple > 1:
            T = int(math.ceil(T_raw / chunk_multiple) * chunk_multiple)
        else:
            T = T_raw

        assert T % (chunk_multiple or 1) == 0, f"T={T} must be multiple of {chunk_multiple}"

        def pad_to(x, value):
            L = x.size(0)
            if L < T:
                return torch.nn.functional.pad(x, (0, T - L), value=value)
            return x[:T]  # truncate if longer

        # Pad inputs, labels, and attention masks
        input_ids = torch.stack([pad_to(x, pad_id) for x in input_ids], dim=0)
        # labels: pad with -100 so cross-entropy ignores them
        labels = torch.stack([pad_to(x, -100) for x in labels], dim=0)
        # attention mask: 1 for real tokens, 0 for pad
        attn = torch.stack([pad_to(x, 0) for x in attn], dim=0).to(torch.long)

        # Your `training_step` expects (idx, targets) when args.my_qa_mask != 1
        if return_mask:
            return input_ids, labels, attn
        else:
            return input_ids, labels

    return _collate
